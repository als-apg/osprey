"""``OspreyPVConnector.reachable``: what the adapter's own channels say about the machine.

The adapter counts its channels that are connected. A drop that takes the count
to zero and holds for the hold-down reads ``no``, ``since`` the drop, stamped
once in epoch seconds; the first channel up reads ``yes``. A release that takes
the count to zero is no evidence and reads ``unknown``. Nothing but the
adapter's own channels decides: no connector evidence is asked, so where no
channel has dropped the statement is ``unknown`` with no reason.

The display-only server report carries the fleet's statement instead:
``reachable`` ``yes`` until the transport is stale (``since`` an ISO string),
and ``writes`` the session's verdict, not the heartbeat claim.

Everything runs in process over :class:`~tests._fake_connector.FakeConnector`
with the adapter's two clocks replaced by one fake, so the hold-down is stepped,
never slept.
"""

from __future__ import annotations

import inspect
import json
import os
from datetime import UTC, datetime
from pathlib import Path

import pytest
from canopy.epics.base import PVConnectionFailed

from osprey.integrations.canopy import backend as backend_module
from osprey.integrations.canopy.backend import (
    REACH_NO,
    REACH_UNKNOWN,
    REACH_YES,
    RENDERED_FOR_ENV,
    OspreyPVConnector,
)
from osprey.interfaces.web_terminal.supervised_child import PREVIOUS_EXIT_ENV
from osprey.runtime import session as session_module
from osprey_connectors import control_context, posture_store, types
from osprey_connectors.control_system import _epics_connector as epics_connector_module
from osprey_connectors.factory import ConnectorFactory
from osprey_connectors.identity import AUDIT_IDENTITY_ENV, TERMINAL_USER_ENV
from osprey_connectors_tests._fake_connector import FakeConnector
from osprey_connectors_tests._harness import isolated_connector_registries

# start() arms the real raw-put block in this process; put back what it patched.
pytestmark = pytest.mark.usefixtures("raw_put_block_restored")

A = "SR:A"
B = "SR:B"
#: Names the fake never connects until restored.
NEVER = ("NEVER:B", "NEVER:C")

HOLD_S = epics_connector_module.STALE_HOLD_DOWN_S

#: Where the fake wall clock starts: an epoch-seconds value, far from the monotonic one.
WALL_START = 1_760_700_000.0


class FakeClock:
    """One fake time for both of the adapter's clocks."""

    def __init__(self) -> None:
        self.now = 1000.0

    def monotonic(self) -> float:
        return self.now

    def time(self) -> float:
        return WALL_START + (self.now - 1000.0)

    def advance(self, seconds: float) -> None:
        self.now += seconds


class _Fake(FakeConnector):
    """The fake, with a ``reach_evidence()`` the adapter must never ask."""

    asked = 0

    def reach_evidence(self):
        type(self).asked += 1
        return None


class _Exited(BaseException):
    def __init__(self, status: int) -> None:
        super().__init__(status)
        self.status = status


def _write_record(state_dir: Path, *, target: str = "va", generation: int = 3) -> None:
    control_context.write_record(
        control_context.ControlContext(target=target, generation=generation, posture={}),
        path=state_dir / control_context.RECORD_FILENAME,
    )
    control_context.invalidate_cache()


@pytest.fixture
def deployment(tmp_path, monkeypatch):
    """A session environment bound to ``va@3`` over :class:`_Fake`; yields the record dir."""
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
        if name.startswith(("OSPREY_CONTROL_TARGET", "EPICS_", "CANOPY_EPICS__")):
            monkeypatch.delenv(name)
    for name in (
        posture_store.CONTROL_CONTEXT_TREE_ENV_VAR,
        posture_store.CONTROL_OWNER_ENV_VAR,
        posture_store.LAUNCH_POSTURE_ENV_VAR,
        "OSPREY_EXECUTION_MODE",
        PREVIOUS_EXIT_ENV,
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(session_module, "MOVE_POLL_S", 0.01)
    monkeypatch.chdir(tmp_path)
    section = {
        "type": types.VIRTUAL_ACCELERATOR,
        "connector": {
            types.VIRTUAL_ACCELERATOR: {"response_delay_ms": 0, "never_connects": list(NEVER)}
        },
    }

    def get_config_value(path, default=None, _config_path=None):
        return section if path == "control_system" else default

    monkeypatch.setattr("osprey_connectors.config.get_config_value", get_config_value)

    def fake_exit(status):
        raise _Exited(status)

    monkeypatch.setattr(backend_module.os, "_exit", fake_exit)
    _Fake.asked = 0
    _write_record(directory)
    monkeypatch.setenv(RENDERED_FOR_ENV, "va@3")
    with isolated_connector_registries():
        ConnectorFactory.register_control_system(types.VIRTUAL_ACCELERATOR, _Fake)
        yield directory
    control_context.invalidate_cache()


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
async def adapter(deployment, clock):  # noqa: ARG001 - the fixture binds the deployment
    connector = OspreyPVConnector(
        connect_report_s=5.0, clock=clock.monotonic, wall_clock=clock.time
    )
    await connector.start()
    try:
        yield connector
    finally:
        await connector.close()


def _fake(adapter: OspreyPVConnector) -> _Fake:
    return adapter._session._connector


def _state(adapter: OspreyPVConnector) -> str:
    return adapter.reachable["state"]


#: The fake-time step a walk reads at; a binary fraction, so steps add up exactly.
STEP_S = 0.125


def _walk(adapter: OspreyPVConnector, clock: FakeClock, seconds: float, step: float = STEP_S):
    """Every statement read while stepping ``seconds`` of fake time, one per ``step``."""
    seen = []
    for _ in range(round(seconds / step)):
        clock.advance(step)
        seen.append(adapter.reachable)
    return seen


async def _subscribe(adapter: OspreyPVConnector, pv: str):
    return await adapter.subscribe(pv, lambda _event: None)


class TestTheShape:
    async def test_no_statement_before_a_session_is_open(self, deployment, clock):  # noqa: ARG002
        connector = OspreyPVConnector(clock=clock.monotonic, wall_clock=clock.time)
        assert connector.reachable is None

    async def test_unknown_at_boot_with_nothing_subscribed(self, adapter):
        assert adapter.reachable == {"state": REACH_UNKNOWN, "since": None, "reason": None}

    async def test_the_first_channel_up_is_yes(self, adapter):
        await _subscribe(adapter, A)
        assert adapter.reachable == {"state": REACH_YES, "since": None, "reason": None}

    async def test_a_subscription_that_never_connects_is_not_yes(self, adapter, clock):
        await _subscribe(adapter, NEVER[0])
        assert {s["state"] for s in _walk(adapter, clock, 30.0)} == {REACH_UNKNOWN}

    async def test_none_again_after_close(self, adapter):
        await _subscribe(adapter, A)
        await adapter.close()
        assert adapter.reachable is None


class TestTheDrop:
    async def test_no_after_the_hold_down_since_the_drop(self, adapter, clock):
        await _subscribe(adapter, A)
        clock.advance(7.0)
        dropped_at = clock.time()
        _fake(adapter).drop(A)
        before = _walk(adapter, clock, HOLD_S - STEP_S)
        assert {s["state"] for s in before} == {REACH_YES}, "a bounce shorter than the hold-down"
        clock.advance(STEP_S)
        assert adapter.reachable == {"state": REACH_NO, "since": dropped_at, "reason": None}

    async def test_since_is_stamped_once_and_never_moves(self, adapter, clock):
        await _subscribe(adapter, A)
        dropped_at = clock.time()
        _fake(adapter).drop(A)
        later = _walk(adapter, clock, 60.0, step=1.0)[round(HOLD_S) :]
        assert {s["state"] for s in later} == {REACH_NO}
        assert {s["since"] for s in later} == {dropped_at}
        assert isinstance(later[0]["since"], float)

    async def test_five_one_second_drops_in_ten_seconds_say_no_no(self, adapter, clock):
        await _subscribe(adapter, A)
        seen = []
        for _ in range(5):
            _fake(adapter).drop(A)
            seen += _walk(adapter, clock, 1.0)
            _fake(adapter).restore(A)
            seen += _walk(adapter, clock, 1.0)
        assert len(seen) == 10 / STEP_S
        assert {s["state"] for s in seen} == {REACH_YES}

    async def test_the_first_channel_back_is_yes(self, adapter, clock):
        await _subscribe(adapter, A)
        _fake(adapter).drop(A)
        clock.advance(HOLD_S + 1)
        assert _state(adapter) == REACH_NO
        _fake(adapter).restore(A)
        assert adapter.reachable == {"state": REACH_YES, "since": None, "reason": None}

    async def test_the_count_is_every_connected_channel(self, adapter, clock):
        await _subscribe(adapter, A)
        await _subscribe(adapter, B)
        _fake(adapter).drop(A)
        assert {s["state"] for s in _walk(adapter, clock, HOLD_S + 1)} == {REACH_YES}
        dropped_at = clock.time()
        _fake(adapter).drop(B)
        clock.advance(HOLD_S)
        assert adapter.reachable["since"] == dropped_at, "since is the drop to zero"

    async def test_two_monitors_of_one_pv_are_one_channel(self, adapter, clock):
        first = await _subscribe(adapter, A)
        await _subscribe(adapter, A)
        await first.close()
        assert _state(adapter) == REACH_YES
        _fake(adapter).drop(A)
        clock.advance(HOLD_S)
        assert _state(adapter) == REACH_NO


class TestTheRelease:
    async def test_releasing_the_last_connected_channel_is_unknown_never_no(self, adapter, clock):
        monitor = await _subscribe(adapter, A)
        await monitor.close()
        assert {s["state"] for s in _walk(adapter, clock, 30.0, step=0.5)} == {REACH_UNKNOWN}

    async def test_a_release_inside_the_hold_down_is_unknown(self, adapter, clock):
        monitor = await _subscribe(adapter, A)
        _fake(adapter).drop(A)
        clock.advance(HOLD_S / 2)
        await monitor.close()
        assert {s["state"] for s in _walk(adapter, clock, 30.0, step=0.5)} == {REACH_UNKNOWN}

    async def test_release_during_the_outage_then_resubscribe_carries_the_since(
        self, adapter, clock
    ):
        """Drop → no; release → unknown; re-subscribe while down → unknown; back up → yes."""
        monitor = await _subscribe(adapter, A)
        dropped_at = clock.time()
        _fake(adapter).drop(A)
        clock.advance(HOLD_S)
        assert adapter.reachable == {"state": REACH_NO, "since": dropped_at, "reason": None}

        await monitor.close()
        released = _walk(adapter, clock, 10.0)
        assert {s["state"] for s in released} == {REACH_UNKNOWN}

        await _subscribe(adapter, A)  # still down: the fake reports nothing
        still_down = _walk(adapter, clock, 30.0)
        assert {s["state"] for s in still_down} == {REACH_UNKNOWN}, "a channel says nothing"
        assert {s["reason"] for s in still_down} == {None}
        _fake(adapter).restore(A)
        assert adapter.reachable == {"state": REACH_YES, "since": None, "reason": None}

    async def test_a_release_that_keeps_a_dropped_channel_held_keeps_the_no(self, adapter, clock):
        first = await _subscribe(adapter, A)
        await _subscribe(adapter, B)
        dropped_at = clock.time()
        _fake(adapter).drop(A)
        _fake(adapter).drop(B)
        clock.advance(HOLD_S)
        await first.close()
        assert adapter.reachable == {"state": REACH_NO, "since": dropped_at, "reason": None}

    async def test_a_never_connected_channel_left_subscribed_is_still_a_release(
        self, adapter, clock
    ):
        """Count 1 by A, B never connected; release A → unknown, and it stays unknown."""
        monitor = await _subscribe(adapter, A)
        await _subscribe(adapter, NEVER[0])
        assert _state(adapter) == REACH_YES
        await monitor.close()
        seen = _walk(adapter, clock, 30.0)
        assert {s["state"] for s in seen} == {REACH_UNKNOWN}
        assert {s["reason"] for s in seen} == {None}


class TestOwnChannelsOnly:
    async def test_after_a_release_nothing_says_no(self, adapter, clock):
        monitor = await _subscribe(adapter, A)
        await monitor.close()
        seen = _walk(adapter, clock, 30.0, step=0.5)
        assert seen[-1] == {"state": REACH_UNKNOWN, "since": None, "reason": None}
        assert {s["state"] for s in seen} == {REACH_UNKNOWN}

    async def test_the_connector_is_never_asked_for_evidence(self, adapter, clock):
        monitor = await _subscribe(adapter, A)
        _fake(adapter).drop(A)
        _walk(adapter, clock, HOLD_S + 1)
        await monitor.close()
        await _subscribe(adapter, NEVER[0])
        _walk(adapter, clock, 30.0)
        assert _Fake.asked == 0

    async def test_a_stale_transport_is_no_since_the_probe(self, adapter, clock):
        await _subscribe(adapter, A)
        probed_at = clock.time()
        adapter._state_stale()
        clock.advance(10.0)
        assert adapter.reachable == {"state": REACH_NO, "since": probed_at, "reason": None}

    async def test_a_stale_transport_keeps_the_since_of_a_no_already_stated(self, adapter, clock):
        await _subscribe(adapter, A)
        dropped_at = clock.time()
        _fake(adapter).drop(A)
        clock.advance(HOLD_S + 5)
        assert adapter.reachable["since"] == dropped_at
        adapter._state_stale()
        assert adapter.reachable == {"state": REACH_NO, "since": dropped_at, "reason": None}


def _report_statement() -> dict:
    """The ``statement`` of this process's display-only server report, as written."""
    path = control_context.report_path(os.getpid())
    assert path is not None
    return json.loads(path.read_text(encoding="utf-8"))["statement"]


class TestTheReportStatement:
    """The display-only report: the fleet's ``reachable`` and the session's verdict."""

    async def test_the_report_states_yes_and_the_sessions_verdict(self, adapter):
        statement = _report_statement()
        assert statement["reachable"] == {"state": "yes", "since": None, "reason": None}
        assert statement["writes"] == adapter._session.writes
        assert set(statement["writes"]) == {"allowed", "position", "reason", "kind"}
        # Not the heartbeat's claim, which is {state, reason, cause}.
        assert set(statement["writes"]) != set(adapter.writes)

    async def test_a_channel_that_drops_never_makes_the_report_say_no(self, adapter, clock):
        await _subscribe(adapter, A)
        _fake(adapter).drop(A)
        clock.advance(HOLD_S + 1)
        assert _state(adapter) == REACH_NO
        adapter.statement_sent()
        assert _report_statement()["reachable"]["state"] == "yes"

    async def test_a_stale_transport_reports_no_since_an_iso_time(self, adapter, clock):
        probed_at = clock.time()
        adapter._state_stale()
        adapter.statement_sent()
        reachable = _report_statement()["reachable"]
        assert reachable == {
            "state": "no",
            "since": datetime.fromtimestamp(probed_at, UTC).isoformat(),
            "reason": None,
        }


class TestTheReportWriter:
    """Only a write the file system refuses is let go; a report that cannot be built raises."""

    async def test_a_refused_file_write_is_logged_and_the_flush_goes_on(
        self, adapter, monkeypatch, caplog
    ):
        def refuse(*_args, **_kwargs):
            raise PermissionError("read-only state dir")

        monkeypatch.setattr(control_context, "write_json_atomic", refuse)
        adapter._statement_sent.clear()

        with caplog.at_level("WARNING", logger=backend_module.logger.name):
            adapter.statement_sent()

        assert adapter._statement_sent.is_set()
        assert "Could not write the display-only server report" in caplog.text

    async def test_a_report_that_cannot_be_built_raises(self, adapter, monkeypatch):
        """The defect the old ``display_only=True`` call hid behind a warning."""

        def wrong_shape(**_kwargs):
            raise TypeError("ServerReport() got an unexpected keyword argument")

        monkeypatch.setattr(control_context, "ServerReport", wrong_shape)

        with pytest.raises(TypeError):
            adapter.statement_sent()


class TestTheSwitch:
    async def test_a_switch_during_an_outage_gives_the_new_target_its_own_since(
        self, deployment, clock
    ):
        """A on va@3 drops to no; the process for the next binding opens during the outage.

        The record names a new binding, which is a new process; it carries
        nothing of the outage it was started into, and its own ``no`` has its
        own drop's ``since``.
        """
        a = OspreyPVConnector(connect_report_s=5.0, clock=clock.monotonic, wall_clock=clock.time)
        await a.start()
        try:
            await _subscribe(a, A)
            a_since = clock.time()
            _fake(a).drop(A)
            clock.advance(HOLD_S + 10)
            assert a.reachable == {"state": REACH_NO, "since": a_since, "reason": None}
        finally:
            await a.close()

        _write_record(deployment, generation=4)
        os.environ[RENDERED_FOR_ENV] = "va@4"
        b = OspreyPVConnector(connect_report_s=5.0, clock=clock.monotonic, wall_clock=clock.time)
        await b.start()
        try:
            assert b.binding == "va@4"
            await _subscribe(b, NEVER[0])
            seen = _walk(b, clock, 30.0)
            assert {s["state"] for s in seen} == {REACH_UNKNOWN}, "nothing carried across"
            await _subscribe(b, A)
            b_since = clock.time()
            _fake(b).drop(A)
            clock.advance(HOLD_S)
            assert b.reachable == {"state": REACH_NO, "since": b_since, "reason": None}
            assert b_since != a_since
        finally:
            await b.close()

    async def test_a_session_that_will_not_open_states_nothing(self, deployment, clock):
        os.environ[RENDERED_FOR_ENV] = "elsewhere@9"
        connector = OspreyPVConnector(clock=clock.monotonic, wall_clock=clock.time)
        with pytest.raises((_Exited, PVConnectionFailed)):
            await connector.start()
        assert connector.reachable is None
        assert deployment.exists()


def test_since_never_comes_from_the_bulk_records_monotonic_down_since():
    """``since`` is epoch seconds stamped by the adapter; nothing reads ``down_since``."""
    source = inspect.getsource(backend_module)
    assert "down_since" not in source
    assert "snapshot(" not in source
