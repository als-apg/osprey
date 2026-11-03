"""CANOPY's PV backend reading through a real control session.

Every test opens a real :class:`~osprey.runtime.session.ControlSession` against
a real control-context record in ``tmp_path``; the backend is built the way
CANOPY builds it, by its ``module:Class`` path. The control system behind it is
the fake connector, registered under the virtual accelerator's type, extended
with the three things CANOPY's conformance kit needs and the plain fake does not
have: an enum channel, a waveform channel, and a name nothing answers for.
"""

from __future__ import annotations

import asyncio
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, ClassVar

import pytest
from canopy.contracts.pv import ValueKind
from canopy.contracts.sample import AlarmSeverity
from canopy.epics.base import (
    PVConnectionFailed,
    PVConnector,
    PVReadTimeout,
)
from canopy.epics.config import EpicsConfig
from canopy.epics.gateway import build_connector
from canopy.epics.testing import conformance

from osprey.integrations.canopy import backend as backend_module
from osprey.integrations.canopy.backend import OspreyPVConnector, PVTargetChanged, PVWriteRefused
from osprey.runtime import session as session_module
from osprey_connectors import control_context, posture_store, types
from osprey_connectors.control_system._base import ChannelMetadata, ChannelValue
from osprey_connectors.factory import ConnectorFactory
from osprey_connectors.identity import AUDIT_IDENTITY_ENV, TERMINAL_USER_ENV
from osprey_connectors.refusals import refusal_phrase
from osprey_connectors_tests._fake_connector import FakeConnector, FakeSubscription
from osprey_connectors_tests._harness import isolated_connector_registries

BACKEND_PATH = "osprey.integrations.canopy.backend:OspreyPVConnector"

#: A deployment whose baseline is the virtual accelerator.
VA_SECTION = {
    "type": types.VIRTUAL_ACCELERATOR,
    "connector": {types.VIRTUAL_ACCELERATOR: {"response_delay_ms": 0}},
}

#: The channels the machine serves, by the conformance kit's kind.
CHANNELS = {
    "scalar": "TEST:SCALAR",
    "enum": "TEST:MODE",
    "waveform": "TEST:WAVEFORM",
    conformance.NEVER: "TEST:NOBODY:HOME",
}
PREFIXES = {"bare": "", "ca": "ca://", "pva": "pva://"}
ENUM_LABELS = ["Off", "Standby", "On"]

SCALAR = CHANNELS["scalar"]
ENUM = CHANNELS["enum"]
WAVEFORM = CHANNELS["waveform"]
NEVER = CHANNELS[conformance.NEVER]

#: A channel that reports every limit and its write access.
LIMITED = "TEST:LIMITED"
LIMITS = {
    "control_low": -10.0,
    "control_high": 10.0,
    "warning_low": -6.0,
    "warning_high": 6.0,
    "alarm_low": -8.0,
    "alarm_high": 8.0,
}


class _Machine(FakeConnector):
    """A fake control system whose channels outlive any one connector.

    The values live on the class, so a test can set one before the session has
    built its connector — as the conformance kit does — and every connected
    instance tells its subscribers when one changes.
    """

    store: ClassVar[dict[str, Any]] = {}
    live: ClassVar[list[_Machine]] = []
    reads: ClassVar[int] = 0
    subscribes: ClassVar[int] = 0
    writes: ClassVar[int] = 0

    @classmethod
    def reset(cls) -> None:
        cls.store = {}
        cls.live = []
        cls.reads = cls.subscribes = cls.writes = 0

    @classmethod
    def set(cls, address: str, value: Any) -> None:
        cls.store[address] = value
        for machine in list(cls.live):
            machine._notify([address])

    @classmethod
    def active(cls) -> int:
        return sum(len(machine._subscriptions) for machine in cls.live)

    async def connect(self, config):
        await super().connect(config)
        type(self).live.append(self)

    async def disconnect(self):
        if self in type(self).live:
            type(self).live.remove(self)
        await super().disconnect()

    def _read_value(self, channel_address):
        now = datetime.now(UTC)
        return ChannelValue(
            value=type(self).store[channel_address],
            timestamp=now,
            metadata=ChannelMetadata(
                units="mA" if channel_address == SCALAR else "",
                precision=3 if channel_address == SCALAR else None,
                description=f"Machine channel {channel_address}",
                enum_labels=list(ENUM_LABELS) if channel_address == ENUM else None,
                timestamp=now,
                **(dict(LIMITS, writable=True) if channel_address == LIMITED else {}),
            ),
        )

    async def read_channel(self, channel_address, timeout=None):  # noqa: ARG002 - the connector read signature
        type(self).reads += 1
        if channel_address not in type(self).store:
            raise TimeoutError(f"{channel_address} did not answer")
        return self._read_value(channel_address)

    async def subscribe(self, channel_address, callback, *, on_connection=None):
        type(self).subscribes += 1
        self._subscription_count += 1
        sub_id = f"machine_{channel_address}_{self._subscription_count}"
        sub = FakeSubscription(channel_address, callback, on_connection)
        self._subscriptions[sub_id] = sub
        if channel_address in type(self).store and self._is_up(channel_address):
            self._report_connection(sub, True)
            self._deliver(sub_id, sub)
        return sub_id

    def _notify(self, channel_addresses):
        super()._notify(address for address in channel_addresses if address in type(self).store)

    async def write_channel(self, *args, **kwargs):
        type(self).writes += 1
        return await super().write_channel(*args, **kwargs)


class _RefusingConnector(FakeConnector):
    """A connector whose control system is not there."""

    async def connect(self, config):  # noqa: ARG002 - the connector connect signature
        raise ConnectionError("no gateway answers")


def write_record(state_dir: Path, *, target="va", generation=3) -> Path:
    path = state_dir / control_context.RECORD_FILENAME
    control_context.write_record(
        control_context.ControlContext(target=target, generation=generation, posture={}),
        path=path,
    )
    control_context.invalidate_cache()
    return path


@pytest.fixture(autouse=True)
def armed(monkeypatch):
    """Every arming of the raw-put block, recorded instead of patching this test process."""
    calls: list[str] = []
    monkeypatch.setattr(backend_module, "_arm_raw_put_block", lambda: calls.append("armed"))
    return calls


@pytest.fixture
def state_dir(tmp_path, monkeypatch):
    """A session process's environment, serving :data:`VA_SECTION`, with no record yet.

    Writes are refused, as the section's tier does not allow them: the reads
    here never write.
    """
    directory = tmp_path / "control_target"
    directory.mkdir()
    config = tmp_path / "config.yml"
    config.write_text("{}\n", encoding="utf-8")
    monkeypatch.setenv(posture_store.CONTROL_CONTEXT_DIR_ENV_VAR, str(directory))
    monkeypatch.setenv(AUDIT_IDENTITY_ENV, "test-operator")
    monkeypatch.delenv(TERMINAL_USER_ENV, raising=False)
    monkeypatch.setenv("OSPREY_CONFIG", str(config))
    monkeypatch.setenv("CONFIG_FILE", str(config))
    for name in list(os.environ):
        if name.startswith(
            ("OSPREY_CONTROL_TARGET", "EPICS_", "CANOPY_EPICS__", "OSPREY_CANOPY_WRITE")
        ):
            monkeypatch.delenv(name)
    for name in (
        posture_store.CONTROL_CONTEXT_TREE_ENV_VAR,
        posture_store.CONTROL_OWNER_ENV_VAR,
        posture_store.LAUNCH_POSTURE_ENV_VAR,
        "OSPREY_EXECUTION_MODE",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(session_module, "MOVE_POLL_S", 0.01)
    monkeypatch.chdir(tmp_path)

    def get_config_value(path, default=None, _config_path=None):
        return VA_SECTION if path == "control_system" else default

    monkeypatch.setattr("osprey_connectors.config.get_config_value", get_config_value)

    def no_exit(status):
        raise AssertionError(f"the backend tried to end the test process with status {status}")

    monkeypatch.setattr(backend_module.os, "_exit", no_exit)
    control_context.invalidate_cache()
    yield directory
    control_context.invalidate_cache()


@pytest.fixture
def machine(monkeypatch, state_dir):
    """A bound deployment: record ``va@3``, rendered for it, served by :class:`_Machine`."""
    write_record(state_dir)
    monkeypatch.setenv(backend_module.RENDERED_FOR_ENV, "va@3")
    _Machine.reset()
    with isolated_connector_registries():
        ConnectorFactory.register_control_system(types.VIRTUAL_ACCELERATOR, _Machine)
        yield _Machine
    _Machine.reset()


def make_connector(**settings) -> PVConnector:
    """A fresh, unstarted backend, loaded the way CANOPY loads it."""
    return build_connector(EpicsConfig(backend=BACKEND_PATH, **settings))


@pytest.fixture
async def connector(machine):
    """A started backend over :class:`_Machine`, with short waits."""
    machine.set(SCALAR, 1.25)
    machine.set(ENUM, 1)
    built = make_connector(ca_timeout_s=0.3, connect_report_s=0.1)
    await built.start()
    yield built
    await built.close()


class _Driver:
    """The conformance kit's handle on :class:`_Machine`."""

    def pv(self, kind: str, form: str) -> str:
        return PREFIXES[form] + CHANNELS[kind]

    async def set_value(self, pv: str, value) -> None:
        for prefix in ("ca://", "pva://"):
            pv = pv.removeprefix(prefix)
        _Machine.set(pv, value)

    async def drop(self) -> None:
        """Take every served channel down on every live machine."""
        for machine in list(_Machine.live):
            for address in list(_Machine.store):
                machine.drop(address)

    async def restore(self) -> None:
        """Bring every served channel back up on every live machine."""
        for machine in list(_Machine.live):
            for address in list(_Machine.store):
                machine.restore(address)


async def _until(predicate, timeout=2.0) -> bool:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not predicate():
        if loop.time() >= deadline:
            return False
        await asyncio.sleep(0.01)
    return True


class TestConformance:
    """The backend behaves as every CANOPY backend must."""

    @pytest.mark.parametrize("case_id", conformance.case_ids())
    async def test_conformance(self, case_id, machine):  # noqa: ARG002 - the fixture binds the deployment
        """One case of CANOPY's backend conformance kit."""
        try:
            await conformance.run_case(case_id, make_connector, _Driver())
        except conformance.CaseSkipped as skip:
            pytest.skip(str(skip))


class TestLoading:
    """How CANOPY gets hold of the backend."""

    def test_canopy_loads_the_backend_by_module_and_class(self, state_dir):  # noqa: ARG002
        """The ``module:Class`` path builds this connector, named ``osprey``."""
        built = make_connector()
        assert isinstance(built, OspreyPVConnector)
        assert built.backend_name == "osprey"
        assert isinstance(built, PVConnector)

    def test_building_the_backend_opens_nothing(self, machine):
        """A connector that was built and never started has no session."""
        make_connector()
        assert machine.live == []

    async def test_a_backend_that_was_not_started_serves_nothing(self, machine):  # noqa: ARG002
        """A read before ``start`` is refused, not answered from thin air."""
        with pytest.raises(PVConnectionFailed):
            await make_connector().get(SCALAR)

    async def test_start_and_close_can_be_repeated(self, machine):
        """Closing disconnects the control system, and a second close is harmless."""
        built = make_connector()
        await built.start()
        await built.start()
        assert len(machine.live) == 1
        await built.close()
        await built.close()
        assert machine.live == []

    async def test_a_fallback_session_rendered_for_no_record_serves(self, state_dir, monkeypatch):  # noqa: ARG002
        """With no record on either side the identities agree, and reads are served."""
        monkeypatch.setenv(backend_module.RENDERED_FOR_ENV, "none")
        _Machine.reset()
        _Machine.set(SCALAR, 4.5)
        with isolated_connector_registries():
            ConnectorFactory.register_control_system(types.VIRTUAL_ACCELERATOR, _Machine)
            built = make_connector()
            await built.start()
            try:
                value = await built.get(SCALAR)
            finally:
                await built.close()
        assert (value.connected, value.value) == (True, 4.5)


class TestGet:
    """One-shot reads."""

    async def test_an_unsubscribed_pv_is_read_once_through_the_session(self, connector, machine):
        """A PV nobody monitors costs one read and no subscription."""
        value = await connector.get(f"ca://{SCALAR}")
        assert (value.pv, value.value, value.value_kind) == (
            f"ca://{SCALAR}",
            1.25,
            ValueKind.SCALAR,
        )
        assert value.severity is AlarmSeverity.NO_ALARM
        assert value.ts_ns > 0
        assert (machine.reads, machine.subscribes) == (1, 0)

    async def test_a_subscribed_pv_is_answered_from_its_subscription(self, connector, machine):
        """A monitored PV is read from what its monitor holds, with no read of its own."""
        subscription = await connector.subscribe(SCALAR, lambda event: None)
        machine.set(SCALAR, 7.0)
        value = await connector.get(SCALAR)
        await subscription.close()
        assert value.value == 7.0
        assert machine.reads == 0

    async def test_a_subscribed_pv_that_has_not_connected_reads_not_connected(self, connector):
        """A monitored PV with no value yet is a not-connected answer, not an error."""
        subscription = await connector.subscribe(NEVER, lambda event: None)
        value = await connector.get(NEVER, timeout=0.05)
        await subscription.close()
        assert (value.connected, value.value, value.pv) == (False, None, NEVER)

    async def test_a_pv_that_does_not_answer_reads_not_connected(self, connector):
        """A failed one-shot read is a not-connected value carrying the reason."""
        value = await connector.get(NEVER)
        assert (value.connected, value.value) == (False, None)
        assert value.severity is AlarmSeverity.INVALID
        assert "did not answer" in value.error

    async def test_a_read_that_outlasts_the_timeout_reads_not_connected(
        self, connector, machine, monkeypatch
    ):
        """A read still pending at the read timeout is abandoned and says how long it waited."""

        async def never(self, channel_address, timeout=None):  # noqa: ARG001 - the connector read signature
            await asyncio.sleep(30)

        monkeypatch.setattr(machine, "read_channel", never)
        value = await connector.get(SCALAR, timeout=0.05)
        assert (value.connected, value.error) == (False, "no answer within 0.05 s")

    async def test_a_long_waveform_is_cut_to_the_configured_length(self, machine):
        """A waveform longer than the limit is truncated and says so."""
        machine.set(CHANNELS["waveform"], [float(n) for n in range(10)])
        built = make_connector(max_array_elements=4)
        await built.start()
        try:
            value = await built.get(CHANNELS["waveform"])
        finally:
            await built.close()
        assert value.value == [0.0, 1.0, 2.0, 3.0]
        assert "truncated to 4 of 10" in value.error


class TestSubscribe:
    """Monitors."""

    async def test_each_spelling_of_a_name_is_its_own_key(self, connector, machine):
        """Three spellings reach one channel name and answer under their own."""
        seen: dict[str, list] = {}
        handles = []
        for prefix in PREFIXES.values():
            pv = prefix + SCALAR
            seen[pv] = []
            handles.append(await connector.subscribe(pv, seen[pv].append))
        machine.set(SCALAR, 9.0)
        for handle in handles:
            await handle.close()
        for pv, events in seen.items():
            assert [(event.pv, event.value) for event in events] == [(pv, 1.25), (pv, 9.0)]

    async def test_two_monitors_of_one_pv_share_one_channel(self, connector, machine):
        """A second monitor of the same name gets the current value without a second channel."""
        first, second = [], []
        one = await connector.subscribe(SCALAR, first.append)
        two = await connector.subscribe(SCALAR, second.append)
        assert machine.subscribes == 1
        assert [event.value for event in second] == [1.25]
        await one.close()
        machine.set(SCALAR, 2.0)
        assert [event.value for event in second] == [1.25, 2.0]
        assert [event.value for event in first] == [1.25]
        await two.close()
        await two.close()
        assert machine.active() == 0

    async def test_a_monitor_that_raises_does_not_starve_the_others(self, connector, machine):
        """One failing callback is logged and skipped."""

        def broken(event):  # noqa: ARG001 - a monitor callback's signature
            raise RuntimeError("bad subscriber")

        seen = []
        one = await connector.subscribe(SCALAR, broken)
        two = await connector.subscribe(SCALAR, seen.append)
        machine.set(SCALAR, 3.0)
        await one.close()
        await two.close()
        assert [event.value for event in seen] == [1.25, 3.0]


class TestOutage:
    """A channel that goes down after it was up."""

    @pytest.mark.parametrize(
        ("address", "kind"),
        [(SCALAR, ValueKind.SCALAR), (ENUM, ValueKind.ENUM), (WAVEFORM, ValueKind.WAVEFORM)],
    )
    async def test_a_dropped_channel_keeps_its_kind(self, connector, machine, address, kind):
        """The not-connected event carries the kind the channel last had, and no value."""
        machine.set(WAVEFORM, [1.0, 2.0])
        events = []
        subscription = await connector.subscribe(address, events.append)
        for live in machine.live:
            live.drop(address)
        held = await connector.get(address)
        await subscription.close()
        assert [(event.connected, event.value_kind) for event in events] == [
            (True, kind),
            (False, kind),
        ]
        assert (events[-1].value, events[-1].severity) == (None, AlarmSeverity.INVALID)
        assert (held.connected, held.value, held.value_kind) == (False, None, kind)

    async def test_a_restored_channel_delivers_again(self, connector, machine):
        """After the outage the channel's current value comes back connected."""
        events = []
        subscription = await connector.subscribe(ENUM, events.append)
        for live in machine.live:
            live.drop(ENUM)
        machine.set(ENUM, 2)
        for live in machine.live:
            live.restore(ENUM)
        await subscription.close()
        assert [(event.connected, event.value) for event in events] == [
            (True, 1),
            (False, None),
            (True, 2),
        ]

    async def test_a_channel_that_never_delivered_reports_a_scalar(self, connector):
        """With no kind known, the not-connected report is a scalar."""
        events = []
        subscription = await connector.subscribe(NEVER, events.append)
        assert await _until(lambda: bool(events))
        await subscription.close()
        assert [(event.connected, event.value_kind) for event in events] == [
            (False, ValueKind.SCALAR)
        ]


class TestDescribe:
    """Static metadata."""

    async def test_a_subscribed_pv_is_described_from_its_subscription(self, connector, machine):
        """The metadata is the monitor's own; no second channel, no read."""
        subscription = await connector.subscribe(f"pva://{ENUM}", lambda event: None)
        meta = await connector.describe(f"pva://{ENUM}")
        assert (meta.pv, meta.value_kind, meta.enum_choices) == (
            f"pva://{ENUM}",
            ValueKind.ENUM,
            ENUM_LABELS,
        )
        assert (machine.subscribes, machine.reads) == (1, 0)
        assert machine.active() == 1
        await subscription.close()

    async def test_an_unsubscribed_pv_is_described_by_a_short_lived_subscription(
        self, connector, machine
    ):
        """Describing a PV nobody monitors leaves no subscription behind."""
        meta = await connector.describe(SCALAR)
        assert (meta.units, meta.precision, meta.value_kind) == ("mA", 3, ValueKind.SCALAR)
        assert meta.description == f"Machine channel {SCALAR}"
        assert meta.enum_choices is None
        assert (machine.subscribes, machine.active()) == (1, 0)

    async def test_the_limits_and_write_access_are_described(self, connector, machine):
        """Control, warning and alarm limits and ``writable`` come through as delivered."""
        machine.set(LIMITED, 2.0)
        meta = await connector.describe(LIMITED)
        assert {name: getattr(meta, name) for name in LIMITS} == LIMITS
        assert meta.writable is True

    async def test_a_channel_that_reports_no_limits_is_described_without_them(self, connector):
        """No limits reported is ``None`` for each, not a made-up range."""
        meta = await connector.describe(SCALAR)
        assert [getattr(meta, name) for name in LIMITS] == [None] * len(LIMITS)
        assert meta.writable is None

    async def test_a_channel_that_delivers_nothing_is_not_described(self, connector, machine):
        """No metadata within the read timeout raises rather than answering empty."""
        with pytest.raises(PVReadTimeout):
            await connector.describe(NEVER)
        assert machine.active() == 0

    async def test_a_monitored_channel_that_delivers_nothing_is_not_described(self, connector):
        """The same holds when a monitor is already waiting on the channel."""
        subscription = await connector.subscribe(NEVER, lambda event: None)
        with pytest.raises(PVReadTimeout):
            await connector.describe(NEVER)
        await subscription.close()


class TestPut:
    """Writes while this login's tier does not allow them: the verdict refuses.

    The write path itself is ``test_backend_write.py``'s.
    """

    async def test_a_write_is_refused_and_reaches_nothing(self, connector, machine):
        """``put`` raises CANOPY's refusal in the verdict's words; the control system is untouched."""
        with pytest.raises(PVWriteRefused) as caught:
            await connector.put(SCALAR, 5.0, binding="va@3")
        assert connector._session.verdict.reason == "tier_never"
        assert caught.value.detail == refusal_phrase("tier_never", "simulator")
        assert machine.writes == 0
        assert machine.store[SCALAR] == 1.25


class TestUnservable:
    """A control target this process cannot serve: every PV is disconnected."""

    @pytest.fixture
    async def unservable(self, state_dir, monkeypatch):
        write_record(state_dir)
        monkeypatch.setenv(backend_module.RENDERED_FOR_ENV, "va@3")
        with isolated_connector_registries():
            ConnectorFactory.register_control_system(types.VIRTUAL_ACCELERATOR, _RefusingConnector)
            built = make_connector(ca_timeout_s=0.3, connect_report_s=5.0)
            await built.start()
            yield built
            await built.close()

    async def test_it_starts_although_the_record_identity_is_not_the_binding_id(self, unservable):
        """The rendered-for check compares records, so an unservable binding still serves."""
        assert unservable._session.binding.id == "unservable"

    async def test_a_read_answers_not_connected(self, unservable):
        """``get`` is a not-connected value, not an exception."""
        value = await unservable.get(f"ca://{SCALAR}")
        assert (value.pv, value.connected, value.value) == (f"ca://{SCALAR}", False, None)

    async def test_a_monitor_reports_not_connected_once_without_waiting(self, unservable):
        """There is nothing to wait for, so the one report comes at once."""
        events = []
        subscription = await unservable.subscribe(SCALAR, events.append)
        assert await _until(lambda: bool(events), timeout=1.0)
        await asyncio.sleep(0.1)
        await subscription.close()
        assert [(event.connected, event.value) for event in events] == [(False, None)]

    async def test_describe_is_refused(self, unservable):
        """No metadata exists for a channel no control system serves."""
        with pytest.raises(PVConnectionFailed):
            await unservable.describe(SCALAR)

    async def test_a_write_is_refused(self, unservable):
        """A binding other than ``unservable`` is a changed target; its own is unservable."""
        with pytest.raises(PVTargetChanged):
            await unservable.put(SCALAR, 1.0, binding="va@3")
        with pytest.raises(PVWriteRefused) as caught:
            await unservable.put(SCALAR, 1.0, binding="unservable")
        assert caught.value.detail == "this control target cannot be served here; nothing written"
