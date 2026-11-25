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

import itertools
import json
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from bluesky import Msg, RunEngine
from bluesky import plan_stubs as bps
from bluesky.utils import FailedStatus
from ophyd_async.core import AsyncStatus
from pydantic import BaseModel, ConfigDict

from osprey.services.bluesky_bridge import qserver_startup, session_upload, write_grant
from osprey.services.bluesky_bridge.devices.connector import (
    PRESS_VALUE,
    ConnectorPositioner,
    ConnectorSettable,
    ConnectorTriggerable,
)
from osprey.services.bluesky_bridge.devices.mock import MockReadable
from osprey.services.bluesky_bridge.plan_fields import MovableChannel, TriggeredChannel
from osprey.services.bluesky_bridge.plan_writes import (
    ChannelWrite,
    UndeclaredWriteError,
    UnstatedWriteError,
    message_may_writes,
    message_writes,
)
from osprey.services.bluesky_bridge.preflight import PositionerStopError
from osprey.services.bluesky_bridge.queue_backend import LANE_ENV
from osprey_connectors import session_hold
from osprey_connectors import write_authorization as wa
from osprey_connectors.command import CommandAct
from osprey_connectors.control_system._base import (
    ChannelWriteResult,
    ControlSystemConnector,
    WriteOutcome,
)
from osprey_connectors.posture_store import RESERVED_OWNER_KWARG
from osprey_connectors_tests._harness import isolate_verdict_environment, write_control_context
from osprey_connectors_tests._harness._control_context_fixtures import FIXTURE_IDENTITY
from osprey_connectors_tests._write_fakes import config_reader
from tests.services.bluesky_bridge._plan_devices import ONE_OF_EACH, shipped_plan_device_classes

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


def _plan_function(plan: Any, device: Any, *, schema: type[BaseModel] = _Params) -> Any:
    spec = SimpleNamespace(
        name="quad_scan", schema=schema, plan=plan, description="", metadata=None
    )
    return qserver_startup._make_plan_function(spec, devices={device.name: device})


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
    monkeypatch.delenv(LANE_ENV, raising=False)
    section = config_reader(SECTION)

    def get_config_value(key: str, default: Any = None) -> Any:
        # The lane's declared target: the one its worker's connector is stamped with.
        return TARGET if key == "services.bluesky.target" else section(key, default)

    monkeypatch.setattr("osprey_connectors.config.get_config_value", get_config_value)
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
    _run_device(plan, ConnectorSettable(connector, CHANNEL, name=DEVICE))


def _run_device(plan: Any, device: Any, *, schema: type[BaseModel] = _Params) -> None:
    RunEngine(context_managers=[])(
        _plan_function(plan, device, schema=schema)(**{RESERVED_OWNER_KWARG: ME})
    )


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


# --- the switch moving while a run waits -------------------------------------


class _SwitchMover(threading.Thread):
    """Moves the switch as soon as the run files its request, then (optionally) approves it."""

    def __init__(self, ledger: Path, root: Path, switch: dict[str, str], *, approve: bool) -> None:
        super().__init__(daemon=True)
        self.ledger = ledger
        self.root = root
        self.switch = switch
        self.approve = approve
        self.seen: list[wa.WriteRequest] = []
        self.done = threading.Event()

    def run(self) -> None:
        while not self.done.is_set() and not self.seen:
            requests, _ = wa.list_requests(self.ledger / wa.REQUESTS_DIR_NAME)
            if requests:
                self.seen.extend(requests)
                write_control_context(self.root, target=TARGET, switch=self.switch)
                if self.approve:
                    wa.approve_request(requests[0], surface="cli")
            time.sleep(0.01)


def _moving(ledger: Path, tmp_path: Path, switch: dict[str, str], *, approve: bool) -> _SwitchMover:
    mover = _SwitchMover(ledger, tmp_path / "root", switch, approve=approve)
    mover.start()
    return mover


def test_turning_writes_off_while_a_run_waits_ends_it_and_an_approval_after_moves_nothing(
    ledger: Path, tmp_path: Path
) -> None:
    # The person stops the machine, then (by mistake, or another person) approves.
    mover = _moving(ledger, tmp_path, {}, approve=True)
    connector = _Connector()
    try:
        with pytest.raises(write_grant.WriteAuthorizationError, match="switch_off"):
            _run(_scan, connector)
    finally:
        mover.done.set()
        mover.join(timeout=2)

    assert len(mover.seen) == 1
    assert connector.writes == []
    requests, _ = wa.list_requests(ledger / wa.REQUESTS_DIR_NAME)
    assert requests == ()  # withdrawn
    assert wa.named_authorization() is None


def test_turning_writes_to_allowed_while_a_run_waits_issues_its_own_grant(
    ledger: Path, tmp_path: Path
) -> None:
    mover = _moving(ledger, tmp_path, {TARGET: "allowed"}, approve=False)
    connector = _Connector()
    try:
        _run(_scan, connector)
    finally:
        mover.done.set()
        mover.join(timeout=2)

    assert connector.writes == [1.0, 2.0, 3.0]
    assert len(mover.seen) == 1
    requests, _ = wa.list_requests(ledger / wa.REQUESTS_DIR_NAME)
    assert requests == ()
    [grant] = [
        wa._parse_grant(path.read_text())
        for path in ledger.glob("*.json")
        if not path.name.endswith(".ledger.json")
    ]
    assert grant.approver.surface == wa.SELF_ISSUE_SURFACE


# --- the cap on replacements binds only what a person is asked ---------------


def _creeping(devices: Any, params: Any) -> Iterator[Any]:
    """Step through the points, then keep going one past wherever the quad reads."""
    quad = devices[params.quad]
    yield from _scan(devices, params)
    for _ in range(write_grant.MAX_EXTENSIONS + 2):
        reading = yield Msg("read", quad)
        if reading is None:
            return
        yield from bps.mv(quad, reading[quad.name]["value"] + 1.0)


def test_at_asked_a_run_stops_asking_after_its_last_replacement(person: _Person) -> None:
    connector = _Connector()

    with pytest.raises(FailedStatus):
        _run(_creeping, connector)

    # Nothing moves during the survey, so its reads say 0.0 and it sees only
    # 1.0 past the scan. For real the quad reads 3.0 and creeps to 4.0, 5.0
    # and 6.0, one replacement each; 7.0 is refused by the connector.
    assert connector.writes == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    assert len(person.seen) == 1 + write_grant.MAX_EXTENSIONS


def test_at_allowed_a_run_replaces_its_own_grant_past_the_cap(
    ledger: Path, tmp_path: Path, person: _Person
) -> None:
    write_control_context(tmp_path / "root", target=TARGET, switch={TARGET: "allowed"})
    connector = _Connector()

    _run(_creeping, connector)

    assert connector.writes == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    assert person.seen == []
    grants = [
        wa._parse_grant(path.read_text())
        for path in ledger.glob("*.json")
        if not path.name.endswith(".ledger.json")
    ]
    assert len(grants) == 1 + write_grant.MAX_EXTENSIONS + 2
    assert {grant.approver.surface for grant in grants} == {wa.SELF_ISSUE_SURFACE}


# --- a boolean is the 0/1 it stands for --------------------------------------


def _screen_in_and_out(devices: Any, params: Any) -> Iterator[Any]:
    """Insert a screen with ``True`` and retract it with ``False``: an enum channel's shape."""
    quad = devices[params.quad]
    for value in (True, False):
        yield from bps.mv(quad, value)


def test_a_boolean_write_is_approved_and_sent_as_its_integer(person: _Person) -> None:
    connector = _Connector()

    _run(_screen_in_and_out, connector)

    assert connector.writes == [1, 0]
    assert all(type(value) is int for value in connector.writes)
    [request] = person.seen
    assert request.channels == {CHANNEL: wa.ChannelRange(0.0, 1.0)}
    assert request.steps == 2


def test_a_value_that_is_no_number_still_ends_the_run_before_it_asks(person: _Person) -> None:
    def _text(devices: Any, params: Any) -> Iterator[Any]:
        yield from bps.mv(devices[params.quad], "IN")

    connector = _Connector()

    with pytest.raises(write_grant.WriteAuthorizationError, match="finite numbers only"):
        _run(_text, connector)

    assert person.seen == []
    assert connector.writes == []


# --- what a dead worker left behind ------------------------------------------


def _file_request(label: str) -> wa.WriteRequest:
    return wa.write_request(
        TARGET,
        {CHANNEL: (1.0, 3.0)},
        steps=3,
        expires_at_ns=time.time_ns() + 600 * 10**9,
        lease_s=60.0,
        label=label,
        owner=ME,
    )


def test_a_new_worker_withdraws_only_the_requests_its_own_holder_left(
    ledger: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from osprey_connectors.identity import AUDIT_IDENTITY_ENV, TERMINAL_USER_ENV

    monkeypatch.delenv(TERMINAL_USER_ENV, raising=False)
    monkeypatch.setenv(AUDIT_IDENTITY_ENV, "bluesky-va-queueserver")
    other_lane = _file_request("bluesky orm on the other lane")
    monkeypatch.setenv(AUDIT_IDENTITY_ENV, "queueserver")
    first = _file_request("bluesky orm, worker died waiting")
    second = _file_request("bluesky orm, worker died waiting again")

    withdrawn = write_grant.withdraw_leftover_requests()

    assert set(withdrawn) == {first.id, second.id}
    requests, _ = wa.list_requests(ledger / wa.REQUESTS_DIR_NAME)
    assert [request.id for request in requests] == [other_lane.id]


def test_without_a_ledger_directory_a_new_worker_withdraws_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(wa.LEDGER_DIR_ENV_VAR, raising=False)
    assert write_grant.withdraw_leftover_requests() == ()


@pytest.mark.usefixtures("ledger")
def test_opening_the_worker_environment_withdraws_what_a_dead_worker_left(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[bool] = []
    monkeypatch.setattr(write_grant, "withdraw_leftover_requests", lambda: calls.append(True))
    device = ConnectorSettable(_Connector(), CHANNEL, name=DEVICE)

    qserver_startup.build_namespace(
        {}, run_engine=RunEngine(context_managers=[]), devices={DEVICE: device}, plans={}
    )

    assert calls == [True]


def test_the_bridge_reports_the_newest_request_of_its_worker(
    ledger: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(wa.REQUESTS_DIR_ENV_VAR, str(ledger / wa.REQUESTS_DIR_NAME))
    monkeypatch.setenv(write_grant.WORKER_HOLDER_ENV, ME)
    _file_request("older")
    time.sleep(0.01)
    newer = _file_request("newer")

    reported = write_grant.awaiting_approval()

    assert reported is not None
    assert reported["id"] == newer.id


# --- the grant names the target the lane declares ----------------------------


def test_the_grant_names_the_target_the_lane_declares(person: _Person) -> None:
    connector = _Connector()
    connector._control_target = None  # no stamp on the device's connector to read

    # Approved, the writes still end at the connector's own guard: it refuses
    # a write from an unstamped connector whatever the grant says.
    with pytest.raises(FailedStatus):
        _run(_scan, connector)

    [request] = person.seen
    assert request.target == TARGET
    assert connector.writes == []


# --- every write a device states is under the grant --------------------------

COMMAND = "SR:DET:ACQUIRE"
STATUS = "SR:DET:STATUS"
START = "SR:QF:START"
STOP = "SR:QF:STOP"
DETECTOR = "detector"


class _CommandConnector(_Connector):
    """A machine with a command: a write to it moves the status off idle and back."""

    def __init__(self) -> None:
        super().__init__()
        self.status = 0
        self.sent: list[tuple[str, Any]] = []
        self.subscribers: dict[str, tuple[str, Callable[[Any], None]]] = {}

    async def read_channel(self, channel_address: str, timeout: float | None = None):
        if channel_address == STATUS:
            return SimpleNamespace(value=self.status)
        return await super().read_channel(channel_address, timeout)

    async def subscribe(self, channel_address, callback, *, on_connection=None):  # noqa: ARG002 - interface signature
        subscription = f"sub-{len(self.subscribers)}"
        self.subscribers[subscription] = (channel_address, callback)
        return subscription

    async def unsubscribe(self, subscription_id):
        self.subscribers.pop(subscription_id, None)

    async def write_channel(self, channel_address, value, timeout=None, confirm=None):  # noqa: ARG002 - interface signature
        # Not through super(): every subclass's write_channel is guarded, and
        # a second guard would judge (and spend) the one write twice.
        self.sent.append((channel_address, value))
        self.writes.append(value)
        result = ChannelWriteResult(
            channel_address=channel_address, value_written=value, outcome=WriteOutcome.CONFIRMED
        )
        if channel_address == COMMAND:
            for status in (1, 0):
                self.status = status
                for address, callback in list(self.subscribers.values()):
                    if address == STATUS:
                        callback(SimpleNamespace(value=status))
        return result


class _TriggerParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    detector: TriggeredChannel = DETECTOR


def _three_triggers(devices: Any, params: Any) -> Iterator[Any]:
    for _ in range(3):
        yield from bps.trigger(devices[params.detector], wait=True)


def _detector(connector: _Connector) -> ConnectorTriggerable:
    return ConnectorTriggerable(connector, COMMAND, STATUS, idle=0, timeout_s=5.0, name=DETECTOR)


def test_a_plan_that_only_triggers_writes_only_once_a_person_approves_its_triggers(
    person: _Person,
) -> None:
    connector = _CommandConnector()

    _run_device(_three_triggers, _detector(connector), schema=_TriggerParams)

    [request] = person.seen
    assert request.steps == 3
    assert request.channels == {COMMAND: wa.ChannelRange(1.0, 1.0)}
    assert connector.sent == [(COMMAND, PRESS_VALUE)] * 3


class _StartingSettable(ConnectorSettable):
    """A setpoint a move writes, then a start command: two writes per ``set``."""

    def set_writes(self, value: float) -> tuple[ChannelWrite, ...]:
        return (ChannelWrite(CHANNEL, value), ChannelWrite(START, PRESS_VALUE))

    @AsyncStatus.wrap
    async def set(self, value: float) -> None:
        for write in self.set_writes(value):
            await self._osprey_connector.write_channel_checked(write.address, write.value)


def test_a_set_that_states_two_writes_spends_two_steps(person: _Person) -> None:
    connector = _Connector()

    _run_device(_scan, _StartingSettable(connector, CHANNEL, name=DEVICE))

    points = _Params().points
    [request] = person.seen  # no replacement
    assert request.steps == 2 * len(points)
    assert request.channels == {
        CHANNEL: wa.ChannelRange(1.0, 3.0),
        START: wa.ChannelRange(1.0, 1.0),
    }
    assert connector.writes == [1.0, 1, 2.0, 1, 3.0, 1]


def test_the_request_lists_every_write_the_approval_view_lists(person: _Person) -> None:
    device = _StartingSettable(_Connector(), CHANNEL, name=DEVICE)
    _moves, _total_moves, writes, total_writes, stop_writes = qserver_startup.walk_launch(
        _scan({DEVICE: device}, _Params())
    )

    _run_device(_scan, device)

    [request] = person.seen
    assert set(request.channels) == {entry["channel"] for entry in writes + stop_writes}
    assert request.steps == total_writes


def test_a_positioner_run_is_surveyed_with_its_start_writes_and_asks_once(
    person: _Person,
) -> None:
    connector = _CommandConnector()
    positioner = ConnectorPositioner(
        connector, CHANNEL, started_by=CommandAct(COMMAND, STATUS, 0, 5.0), name=DEVICE
    )

    _run_device(_scan, positioner)

    points = _Params().points
    [request] = person.seen  # neither refused nor replaced for its start writes
    assert request.channels == {
        CHANNEL: wa.ChannelRange(1.0, 3.0),
        COMMAND: wa.ChannelRange(1.0, 1.0),
    }
    assert request.steps == 2 * len(points)
    assert "if stopped mid-move" not in request.label
    assert connector.sent == [
        write for point in points for write in ((CHANNEL, point), (COMMAND, PRESS_VALUE))
    ]


class _StoppingSettable(ConnectorSettable):
    """A setpoint whose ``stop()`` writes a stop command while a move is unfinished."""

    def __init__(self, connector: Any) -> None:
        super().__init__(connector, CHANNEL, name=DEVICE)
        self.moved = False

    def set(self, value: float) -> Any:
        self.moved = True
        return super().set(value)

    def stop_writes(self) -> tuple[ChannelWrite, ...]:
        return (ChannelWrite(STOP, PRESS_VALUE),)

    async def stop(self, *, success: bool = False) -> None:  # noqa: ARG002 - the bluesky verb signature
        if self.moved:
            self.moved = False
            await self._osprey_connector.write_channel_checked(STOP, PRESS_VALUE)


def _scan_then_stop(devices: Any, params: Any) -> Iterator[Any]:
    yield from _scan(devices, params)
    yield Msg("stop", devices[params.quad])


def test_a_stop_a_device_states_is_in_the_grant_and_held_back_from_its_moves(
    person: _Person,
) -> None:
    connector = _Connector()

    _run_device(_scan_then_stop, _StoppingSettable(connector))

    [request] = person.seen  # the stop is written under the grant; nobody is asked again
    assert request.channels == {
        CHANNEL: wa.ChannelRange(1.0, 3.0),
        STOP: wa.ChannelRange(1.0, 1.0),
    }
    assert request.steps == 3 + 1
    assert "+ 1 if stopped mid-move" in request.label
    assert connector.writes == [1.0, 2.0, 3.0, PRESS_VALUE]


def test_a_write_past_the_survey_extends_rather_than_spend_the_stop_reserve(
    person: _Person,
) -> None:
    builds = itertools.count()

    def moves(devices: Any, params: Any, *, one_more: bool) -> Iterator[Any]:
        yield from _scan(devices, params)
        if one_more:
            yield from bps.mv(devices[params.quad], 2.0)
        yield Msg("stop", devices[params.quad])

    def one_more_move_for_real(devices: Any, params: Any) -> Iterator[Any]:
        # The wrapper builds the run's plan first and the survey's copy second.
        return moves(devices, params, one_more=next(builds) == 0)

    connector = _Connector()

    _run_device(one_more_move_for_real, _StoppingSettable(connector))

    first, second = person.seen
    assert first.steps == 3 + 1
    # The fourth move finds every step spent or held back: the replacement
    # carries the held-back stop step forward, and the stop goes out under it.
    assert "auth_exhausted" in second.label
    assert second.steps == 1 + write_grant.EXTRA_STEPS_FLOOR
    assert STOP in second.channels
    assert connector.writes == [1.0, 2.0, 3.0, 2.0, PRESS_VALUE]


def test_a_device_that_can_stop_but_states_no_stop_writes_is_refused_before_anybody_is_asked(
    person: _Person,
) -> None:
    class _SilentStop(ConnectorSettable):
        async def stop(self, *, success: bool = False) -> None:  # noqa: ARG002 - the bluesky verb signature
            raise AssertionError("the launch is refused before anything moves")

    connector = _Connector()

    with pytest.raises(UnstatedWriteError, match="does not state the channels its stop writes"):
        _run_device(_scan, _SilentStop(connector, CHANNEL, name=DEVICE))

    assert person.seen == []
    assert connector.writes == []


@pytest.mark.parametrize("word", ["GO", True])
def test_a_command_write_of_a_value_that_is_no_number_ends_the_run_before_it_asks(
    person: _Person, word: Any
) -> None:
    class _WordCommand(ConnectorTriggerable):
        def trigger_writes(self) -> tuple[ChannelWrite, ...]:
            return (ChannelWrite(COMMAND, word),)

    connector = _CommandConnector()
    detector = _WordCommand(connector, COMMAND, STATUS, idle=0, timeout_s=5.0, name=DETECTOR)

    with pytest.raises(write_grant.WriteAuthorizationError, match="finite numbers only"):
        _run_device(_three_triggers, detector, schema=_TriggerParams)

    assert person.seen == []
    assert connector.sent == []


# --- a launch refused for a write asks nobody first --------------------------

_PLAN_SOURCE = """PLAN_METADATA = {{"name": "quad_scan", "writes": {writes}}}

import itertools

from bluesky import plan_stubs as bps
from pydantic import BaseModel, ConfigDict

from osprey.services.bluesky_bridge.plan_fields import MovableChannel

_BUILDS = itertools.count()


class PARAMS(BaseModel):
    model_config = ConfigDict(extra="forbid")

    quad: MovableChannel = {device!r}
    points: list[float] = [1.0, 2.0, 3.0]


def _scan(devices, params, moves):
    if moves:
        for point in params.points:
            yield from bps.mv(devices[params.quad], point)
    yield from bps.null()

"""

_SCAN = """
def build_plan(devices, params):
    return _scan(devices, params, True)
"""

_MOVES_ONLY_FOR_REAL = """
def build_plan(devices, params):
    # The wrapper builds the run's plan first and the survey's copy second.
    return _scan(devices, params, next(_BUILDS) == 0)
"""


def _wrapped(kind: str, body: str, device: Any, *, writes: bool) -> Any:
    """The plan *body* as the catalog or the session wrapper installs it."""
    source = _PLAN_SOURCE.format(writes=writes, device=DEVICE) + body
    if kind == "session":
        namespace: dict[str, Any] = {"__name__": "__main__", DEVICE: device}
        return session_upload.install_session_plan(namespace, "quad_scan", source)
    module: dict[str, Any] = {"__name__": "catalog_plan"}
    exec(compile(source, "<catalog plan>", "exec", dont_inherit=True), module)
    spec = SimpleNamespace(
        name="quad_scan",
        schema=module["PARAMS"],
        plan=module["build_plan"],
        description="",
        metadata=SimpleNamespace(writes=writes),
    )
    return qserver_startup._make_plan_function(spec, devices={DEVICE: device})


def _launch(plan_function: Any) -> None:
    RunEngine(context_managers=[])(plan_function(**{RESERVED_OWNER_KWARG: ME}))


def _nothing_was_asked(ledger: Path, person: _Person, connector: _Connector) -> None:
    assert person.seen == []
    requests, _ = wa.list_requests(ledger / wa.REQUESTS_DIR_NAME)
    assert requests == ()
    assert connector.writes == []


@pytest.mark.parametrize("kind", ["catalog", "session"])
def test_both_wrappers_refuse_an_undeclared_write_before_anybody_is_asked(
    ledger: Path, person: _Person, kind: str
) -> None:
    connector = _Connector()
    device = ConnectorSettable(connector, CHANNEL, name=DEVICE)

    with pytest.raises(UndeclaredWriteError):
        _launch(_wrapped(kind, _SCAN, device, writes=False))

    _nothing_was_asked(ledger, person, connector)


@pytest.mark.parametrize("kind", ["catalog", "session"])
def test_an_undeclared_write_the_survey_did_not_see_is_refused_before_anybody_is_asked(
    ledger: Path, person: _Person, kind: str
) -> None:
    connector = _Connector()
    device = ConnectorSettable(connector, CHANNEL, name=DEVICE)

    with pytest.raises(UndeclaredWriteError):
        _launch(_wrapped(kind, _MOVES_ONLY_FOR_REAL, device, writes=False))

    _nothing_was_asked(ledger, person, connector)


class _Unstated(ConnectorSettable):
    """A setpoint that can be set and states nothing about what that writes."""

    set_writes = None


@pytest.mark.parametrize("kind", ["catalog", "session"])
def test_a_device_that_does_not_state_its_writes_is_refused_before_anybody_is_asked(
    ledger: Path, person: _Person, kind: str
) -> None:
    connector = _Connector()
    device = _Unstated(connector, CHANNEL, name=DEVICE)

    with pytest.raises(UnstatedWriteError):
        _launch(_wrapped(kind, _SCAN, device, writes=True))

    _nothing_was_asked(ledger, person, connector)


# --- every shipped plan device class, surveyed -------------------------------


def test_every_shipped_plan_device_class_has_a_builder() -> None:
    assert set(ONE_OF_EACH) == set(shipped_plan_device_classes())


def _surveyed(messages: list[Any]) -> write_grant.WriteSurvey:
    survey = write_grant.survey_writes(lambda: iter(messages))
    with pytest.raises(StopIteration) as done:
        next(survey)  # nothing in these plans asks the RunEngine for an answer
    return done.value.value


@pytest.mark.parametrize("cls", list(ONE_OF_EACH), ids=lambda cls: cls.__name__)
def test_every_shipped_plan_device_class_is_surveyed_with_all_its_stated_writes(
    cls: type,
) -> None:
    device = ONE_OF_EACH[cls](_Connector())
    verbs = (("set", (1.0,)), ("trigger", ()), ("stop", ()))
    messages = [
        Msg(verb, device, *args) for verb, args in verbs if callable(getattr(device, verb, None))
    ]
    will = [write for msg in messages for write in message_writes(msg)]
    may = [write for msg in messages for write in message_may_writes(msg)]

    survey = _surveyed(messages)

    assert set(survey.ranges) == {write.address for write in will + may}
    assert survey.steps == len(will)
    assert survey.reserve == len(may)
    if cls is MockReadable:
        assert survey.ranges == {}
        assert survey.steps == 0


# --- a positioner left moving is stopped under the run's grant ---------------

#: The act's bound of the stuck stage: how long its move waits for the status to return.
STUCK_TIMEOUT_S = 0.2


class _StuckStage(_CommandConnector):
    """A stage whose start command moves the status off idle, and never back."""

    async def write_channel(self, channel_address, value, timeout=None, confirm=None):  # noqa: ARG002 - interface signature
        self.sent.append((channel_address, value))
        self.writes.append(value)
        result = ChannelWriteResult(
            channel_address=channel_address, value_written=value, outcome=WriteOutcome.CONFIRMED
        )
        if channel_address == COMMAND:
            self.status = 1
            for address, callback in list(self.subscribers.values()):
                if address == STATUS:
                    callback(SimpleNamespace(value=1))
        return result


def _stuck_positioner(connector: _StuckStage) -> ConnectorPositioner:
    return ConnectorPositioner(
        connector,
        CHANNEL,
        started_by=CommandAct(COMMAND, STATUS, 0, STUCK_TIMEOUT_S),
        stopped_by=STOP,
        name=DEVICE,
    )


def _stopped_under_one_grant(ledger: Path, person: _Person, connector: _StuckStage) -> None:
    """One request, and the stop spent a step of its grant: it went out before the release."""
    [request] = person.seen  # nobody is asked again for the stop
    assert request.channels[STOP] == wa.ChannelRange(1.0, 1.0)
    assert "+ 1 if stopped mid-move" in request.label
    assert connector.sent == [(CHANNEL, 1.0), (COMMAND, PRESS_VALUE), (STOP, PRESS_VALUE)]
    spent = wa._parse_ledger(json.loads((ledger / f"{request.id}.ledger.json").read_text()))
    assert [step.channels for step in spent.steps] == [(CHANNEL,), (COMMAND,), (STOP,)]


def test_a_failed_move_is_stopped_under_the_runs_grant(ledger: Path, person: _Person) -> None:
    connector = _StuckStage()

    with pytest.raises(FailedStatus) as failure:
        _run_device(_scan, _stuck_positioner(connector))

    assert not isinstance(failure.value, PositionerStopError)
    _stopped_under_one_grant(ledger, person, connector)


@pytest.mark.parametrize("kind", ["catalog", "session"])
def test_both_wrappers_stop_an_unfinished_move_before_the_grant_is_released(
    ledger: Path, person: _Person, kind: str
) -> None:
    connector = _StuckStage()

    with pytest.raises(FailedStatus):
        _launch(_wrapped(kind, _SCAN, _stuck_positioner(connector), writes=True))

    _stopped_under_one_grant(ledger, person, connector)
