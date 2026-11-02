"""The control session's watches, its exit status and its cached verdict.

:meth:`~osprey.runtime.session.ControlSession.wait_stale` passes through to the
connector and never returns where the connector cannot tell. The session's
``writes`` statement reads an execution verdict cached at open and refreshed
when :meth:`~osprey.runtime.session.ControlSession.wait_moved` sees the
record's or the session's hold record's file signature change, and at every
write's admission. Every test opens a
real session against a real control-context record in ``tmp_path``; the fake
connector is registered under the virtual accelerator's type inside
``isolated_connector_registries``, and its ``go_stale()`` hook stands in for a
stale transport. The fixtures are copied from ``test_control_session.py`` (no
conftest in ``tests/runtime``).
"""

from __future__ import annotations

import asyncio
import os
import time
from collections.abc import Mapping
from pathlib import Path

import pytest

from osprey.runtime import session as session_module
from osprey.runtime.session import (
    EXIT_BINDING_MOVED,
    EXIT_START_REFUSED,
    EXIT_TRANSPORT_STALE,
    ControlSession,
)
from osprey_connectors import control_context, posture_store, session_hold, types
from osprey_connectors.control_system._va_in_process_connector import VAInProcessConnector
from osprey_connectors.factory import ConnectorFactory
from osprey_connectors.identity import AUDIT_IDENTITY_ENV, TERMINAL_USER_ENV
from osprey_connectors_tests._fake_connector import FakeConnector
from osprey_connectors_tests._harness import isolated_connector_registries
from osprey_connectors_tests._harness._control_context_fixtures import _set_switch_to

#: A deployment whose baseline is the virtual accelerator.
VA_SECTION = {
    "type": types.VIRTUAL_ACCELERATOR,
    "connector": {types.VIRTUAL_ACCELERATOR: {"response_delay_ms": 0}},
}

#: A stand-in with no simulator beside it and no real machine named: ``live`` has no answer.
STANDIN_ONLY_SECTION = {"type": types.LIVE_STANDIN, "connector": {types.LIVE_STANDIN: {}}}

#: How long a watch that must never return is given to return anyway.
NEVER_S = 3.0


class _SpyFake(FakeConnector):
    """A fake that counts every ``wait_stale`` it was asked."""

    asked = 0

    async def wait_stale(self) -> None:
        type(self).asked += 1
        await super().wait_stale()


class _NoStaleHook:
    """A connector-shaped object that predates ``wait_stale``."""


def _serve(monkeypatch, section):
    def get_config_value(path, default=None, _config_path=None):
        return section if path == "control_system" else default

    monkeypatch.setattr("osprey_connectors.config.get_config_value", get_config_value)


def write_record(state_dir: Path, *, target="va", generation=3, switch=None) -> Path:
    """Write the record and make its switch exactly *switch* (default: *target* on)."""
    path = state_dir / control_context.RECORD_FILENAME
    control_context.write_record(
        control_context.ControlContext(target=target, generation=generation), path=path
    )
    control_context.invalidate_cache()
    if switch is None:
        wanted = dict.fromkeys({target} & set(types.CONTROL_TARGETS), "allowed")
    elif isinstance(switch, Mapping):
        wanted = dict(switch)
    else:
        wanted = dict.fromkeys(switch, "allowed")
    _set_switch_to(path, wanted)
    control_context.invalidate_cache()
    return path


@pytest.fixture
def state_dir(tmp_path, monkeypatch):
    """A session process's environment: the bound record directory and nothing stale."""
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
        if name.startswith("OSPREY_CONTROL_TARGET") or name.startswith("EPICS_"):
            monkeypatch.delenv(name)
    for name in (
        posture_store.CONTROL_CONTEXT_TREE_ENV_VAR,
        posture_store.CONTROL_OWNER_ENV_VAR,
        posture_store.LAUNCH_POSTURE_ENV_VAR,
        "OSPREY_EXECUTION_MODE",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(session_module, "MOVE_POLL_S", 0.01)
    control_context.invalidate_cache()
    yield directory
    control_context.invalidate_cache()


@pytest.fixture
def mock_va(monkeypatch, state_dir):
    """Serve :data:`VA_SECTION` and build a spying fake for the virtual accelerator."""
    _serve(monkeypatch, VA_SECTION)
    _SpyFake.asked = 0
    with isolated_connector_registries():
        ConnectorFactory.register_control_system(types.VIRTUAL_ACCELERATOR, _SpyFake)
        yield state_dir


async def _opened() -> ControlSession:
    return await ControlSession.open(surface="panel")


async def _never_returns(waiter: asyncio.Future) -> None:
    """Assert ``waiter`` stays pending for :data:`NEVER_S`, then cancel it."""
    await asyncio.sleep(NEVER_S)
    assert not waiter.done()
    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter


class TestExitStatus:
    def test_transport_stale_is_76_beside_75_and_78(self):
        assert EXIT_TRANSPORT_STALE == 76
        assert EXIT_BINDING_MOVED == 75
        assert EXIT_START_REFUSED == 78

    def test_it_is_exported(self):
        assert "EXIT_TRANSPORT_STALE" in session_module.__all__


class TestWaitStale:
    async def test_it_returns_once_the_connector_goes_stale(self, mock_va):
        write_record(mock_va, target="va", generation=3)
        session = await _opened()
        waiter = asyncio.ensure_future(session.wait_stale())

        await asyncio.sleep(0.1)
        assert not waiter.done()
        session._connector.go_stale()

        await asyncio.wait_for(waiter, timeout=5)
        assert _SpyFake.asked == 1
        await session.close()

    async def test_a_fake_that_never_goes_stale_never_returns(self, mock_va):
        write_record(mock_va, target="va", generation=3)
        session = await _opened()
        await _never_returns(asyncio.ensure_future(session.wait_stale()))
        await session.close()

    async def test_an_unservable_session_never_returns(self, state_dir, monkeypatch):
        _serve(monkeypatch, STANDIN_ONLY_SECTION)
        write_record(state_dir, target="live", generation=2)
        session = await _opened()
        assert session._connector is None

        await _never_returns(asyncio.ensure_future(session.wait_stale()))
        await session.close()

    async def test_a_connector_without_the_hook_never_returns(self, mock_va):
        write_record(mock_va, target="va", generation=3)
        session = await _opened()
        connector, session._connector = session._connector, _NoStaleHook()
        try:
            await _never_returns(asyncio.ensure_future(session.wait_stale()))
        finally:
            session._connector = connector
        await session.close()


class TestConnectorDefaults:
    """Where staleness cannot occur, the connector's own watch never returns."""

    async def test_the_in_process_simulator_never_returns(self):
        await _never_returns(asyncio.ensure_future(VAInProcessConnector().wait_stale()))


class _Counted:
    """Counts the session's record and write-state reads by patching them in place."""

    def __init__(self, monkeypatch):
        self.records = 0
        self.verdicts = 0
        read_record = session_module._read_record
        verdict = session_module._verdict

        def counted_record():
            self.records += 1
            return read_record()

        def counted_verdict(target, session_keys=None):
            self.verdicts += 1
            return verdict(target, session_keys)

        monkeypatch.setattr(session_module, "_read_record", counted_record)
        monkeypatch.setattr(session_module, "_verdict", counted_verdict)

    @property
    def reads(self) -> int:
        return self.records + self.verdicts


async def _until(predicate, timeout: float = 2.0) -> float:
    """Wait until ``predicate()`` holds; return how long that took."""
    start = time.monotonic()
    while not predicate():
        assert time.monotonic() - start < timeout, "the condition never held"
        await asyncio.sleep(0.002)
    return time.monotonic() - start


async def _cancelled(waiter: asyncio.Future) -> None:
    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter


#: The switch with the virtual accelerator off: its writes are off for this login.
NARROWED: tuple[str, ...] = ()

#: :data:`VA_SECTION` with the write tier allowed.
ARMED_SECTION = {
    "type": types.VIRTUAL_ACCELERATOR,
    "writes": "allowed",
    "connector": {types.VIRTUAL_ACCELERATOR: {"response_delay_ms": 0}},
}


@pytest.fixture
def armed_va(monkeypatch, mock_va):
    """:func:`mock_va` serving :data:`ARMED_SECTION`, so the tier passes."""
    _serve(monkeypatch, ARMED_SECTION)
    return mock_va


ON = {"allowed": True, "position": "allowed", "reason": None, "kind": "simulator"}


def _off(reason: str) -> dict:
    return {"allowed": False, "position": None, "reason": reason, "kind": "simulator"}


class TestCachedVerdict:
    """``writes`` reads the verdict the session holds; ``wait_moved`` refreshes it."""

    async def test_a_permitted_session_is_allowed_at_its_position(self, armed_va):
        write_record(armed_va)
        session = await _opened()

        assert session.writes == ON
        assert session.verdict.allowed is True
        assert session.target_moving is False
        await session.close()

    async def test_a_session_opened_at_asked_says_so(self, armed_va):
        write_record(armed_va, switch={"va": "asked"})
        session = await _opened()

        assert session.writes == {**ON, "position": "asked"}
        await session.close()

    async def test_a_session_opened_while_narrowed_names_the_switch(self, armed_va):
        write_record(armed_va, switch=NARROWED)
        session = await _opened()

        assert session.writes == _off("switch_off")
        await session.close()

    async def test_stop_writes_leaves_the_statement_as_read(self, armed_va):
        write_record(armed_va)
        session = await _opened()
        session.stop_writes()

        assert session.writes == ON
        await session.close()

    async def test_reading_writes_reads_no_file(self, armed_va, monkeypatch):
        write_record(armed_va)
        session = await _opened()
        counted = _Counted(monkeypatch)

        for _ in range(5):
            assert session.writes == ON

        assert counted.reads == 0
        await session.close()

    async def test_an_unchanged_tick_reads_no_record(self, armed_va, monkeypatch):
        write_record(armed_va)
        session = await _opened()
        counted = _Counted(monkeypatch)

        waiter = asyncio.ensure_future(session.wait_moved())
        await asyncio.sleep(20 * session_module.MOVE_POLL_S)

        assert counted.reads == 0
        assert not waiter.done()
        await _cancelled(waiter)
        await session.close()

    async def test_a_narrowing_turning_unreadable_shows_without_a_restart(
        self, armed_va, monkeypatch
    ):
        path = write_record(armed_va, switch=NARROWED)
        session = await _opened()
        assert session.writes == _off("switch_off")
        waiter = asyncio.ensure_future(session.wait_moved())
        await asyncio.sleep(3 * session_module.MOVE_POLL_S)
        counted = _Counted(monkeypatch)

        def unreadable(_section, target, **_kwargs):
            # What the verdict answers for a record it cannot read; it never raises.
            return posture_store.WriteVerdict(
                allowed=False, reason="record_unavailable", target=target, kind="simulator"
            )

        monkeypatch.setattr(posture_store, "write_verdict", unreadable)
        write_record(armed_va, switch=NARROWED)
        assert path.exists()
        took = await _until(lambda: session.writes["reason"] == "record_unavailable")

        assert session.writes == _off("record_unavailable")
        assert took < 1.0
        assert counted.records == 1
        assert counted.verdicts == 1
        await asyncio.sleep(10 * session_module.MOVE_POLL_S)
        assert counted.reads == 2, "an unchanged tick after the change read again"
        assert not waiter.done(), "an unreadable state of a narrowed session is not a move"
        await _cancelled(waiter)
        await session.close()

    async def _switched_in_place(self, armed_va, session, **record) -> None:
        """Flip the switch under a watching session: it shows in place and is no move."""
        waiter = asyncio.ensure_future(session.wait_moved())
        await asyncio.sleep(3 * session_module.MOVE_POLL_S)
        expected = dict(session.writes)
        write_record(armed_va, **record)
        await _until(lambda: session.writes != expected)
        await asyncio.sleep(3 * session_module.MOVE_POLL_S)
        assert not waiter.done(), "a switch change is not a move"
        assert session.target_moving is False
        await _cancelled(waiter)

    async def test_off_flipped_to_permitted_reads_allowed_in_place(self, armed_va):
        write_record(armed_va, switch=NARROWED)
        session = await _opened()

        await self._switched_in_place(armed_va, session)

        assert session.writes == ON
        await session.close()

    async def test_on_flipped_to_narrowed_names_the_switch_in_place(self, armed_va):
        write_record(armed_va)
        session = await _opened()

        await self._switched_in_place(armed_va, session, switch=NARROWED)

        assert session.writes == _off("switch_off")
        await session.close()

    async def test_allowed_moved_to_asked_reads_asked_in_place(self, armed_va):
        write_record(armed_va)
        session = await _opened()

        await self._switched_in_place(armed_va, session, switch={"va": "asked"})

        assert session.writes == {**ON, "position": "asked"}
        await session.close()

    async def test_a_record_naming_another_binding_marks_the_target_moving(self, armed_va):
        write_record(armed_va, generation=3)
        session = await _opened()
        waiter = asyncio.ensure_future(session.wait_moved())
        await asyncio.sleep(3 * session_module.MOVE_POLL_S)

        write_record(armed_va, generation=4)
        await asyncio.wait_for(waiter, timeout=5)

        assert session.target_moving is True
        await session.close()

    async def test_a_move_before_the_watch_started_returns_at_once(self, armed_va):
        write_record(armed_va, generation=3)
        session = await _opened()
        write_record(armed_va, generation=4)

        await asyncio.wait_for(session.wait_moved(), timeout=5)

        assert session.target_moving is True
        await session.close()

    async def test_a_failed_watch_marks_the_target_moving(self, armed_va, monkeypatch):
        write_record(armed_va)
        session = await _opened()

        def broken(_path):
            raise OSError("the record cannot be stat'ed")

        monkeypatch.setattr(control_context, "file_signature", broken)
        with pytest.raises(OSError):
            await asyncio.wait_for(session.wait_moved(), timeout=5)

        assert session.target_moving is True
        await session.close()

    async def test_a_cancelled_watch_leaves_the_target_standing(self, armed_va):
        write_record(armed_va)
        session = await _opened()
        waiter = asyncio.ensure_future(session.wait_moved())
        await asyncio.sleep(3 * session_module.MOVE_POLL_S)

        await _cancelled(waiter)

        assert session.target_moving is False
        assert session.writes == ON
        await session.close()

    async def test_moved_still_reads_the_record_itself(self, armed_va, monkeypatch):
        write_record(armed_va)
        session = await _opened()
        counted = _Counted(monkeypatch)

        assert session.moved() is False

        assert counted.records == 1
        assert counted.verdicts == 1
        await session.close()


class TestSessionHold:
    """A hold on this process's session pauses writes in place; it never moves the binding."""

    @pytest.fixture
    def held_session(self, tmp_path, monkeypatch):
        root = tmp_path / "holds"
        monkeypatch.setenv(session_hold.HOLD_DIR_ENV_VAR, str(root))
        monkeypatch.setenv(session_module.SESSION_KEY_ENV_VAR, "alice-1")
        # A session with a key but no record is held; the terminal's launcher
        # writes the record when it stamps the key.
        session_hold.ensure_session("alice-1", root=root)
        return root

    @staticmethod
    def _raise(root: Path) -> None:
        assert session_hold.open_flag(
            "alice-1",
            flag_id="flag-1",
            source="watchdog:limits",
            reason="a limit was refused",
            prompt_ref="turn-7",
            prompt_excerpt="set the current to 900",
            root=root,
        )

    async def test_a_hold_raised_before_open_is_read_at_open(self, armed_va, held_session):
        write_record(armed_va)
        self._raise(held_session)
        session = await _opened()

        assert session.writes == _off("session_held")
        assert "watchdog:limits" in session.verdict.detail
        assert session.moved() is False
        await session.close()

    async def test_a_hold_raised_under_the_watch_shows_without_a_move(self, armed_va, held_session):
        write_record(armed_va)
        session = await _opened()
        assert session.writes == ON
        waiter = asyncio.ensure_future(session.wait_moved())
        await asyncio.sleep(3 * session_module.MOVE_POLL_S)

        self._raise(held_session)
        took = await _until(lambda: session.writes["reason"] == "session_held")

        assert took < 1.0
        assert not waiter.done(), "a hold is not a move"
        session_hold.answer_flags(
            "alice-1", ["flag-1"], answer="continue", operator="alice", root=held_session
        )
        await _until(lambda: session.writes == ON)
        assert not waiter.done()
        await _cancelled(waiter)
        await session.close()


class TestSessionKeys:
    """A process serving several sessions is held while any one of them is."""

    @pytest.fixture
    def holds(self, tmp_path, monkeypatch):
        root = tmp_path / "holds"
        monkeypatch.setenv(session_hold.HOLD_DIR_ENV_VAR, str(root))
        monkeypatch.delenv(session_module.SESSION_KEY_ENV_VAR, raising=False)
        for key in ("alice-1", "alice-2"):
            session_hold.ensure_session(key, root=root)
        return root

    @staticmethod
    def _raise(root: Path, key: str) -> None:
        assert session_hold.open_flag(
            key,
            flag_id="flag-1",
            source="watchdog:limits",
            reason="a limit was refused",
            prompt_ref="turn-7",
            prompt_excerpt="set the current to 900",
            root=root,
        )

    @staticmethod
    async def _opened_with(keys: list[str]) -> ControlSession:
        return await ControlSession.open(surface="panel", session_keys=lambda: list(keys))

    async def test_a_hold_on_any_named_session_holds_the_writes(self, armed_va, holds):
        write_record(armed_va)
        self._raise(holds, "alice-2")
        session = await self._opened_with(["alice-1", "alice-2"])

        assert session.writes == _off("session_held")
        assert session.moved() is False
        await session.close()

    async def test_no_named_session_is_no_hold(self, armed_va, holds):
        write_record(armed_va)
        self._raise(holds, "alice-2")
        session = await self._opened_with([])

        assert session.writes == ON
        await session.close()

    async def test_a_held_session_joining_shows_under_the_watch(self, armed_va, holds):
        write_record(armed_va)
        self._raise(holds, "alice-2")
        keys = ["alice-1"]
        session = await self._opened_with(keys)
        assert session.writes == ON
        waiter = asyncio.ensure_future(session.wait_moved())
        await asyncio.sleep(3 * session_module.MOVE_POLL_S)

        keys.append("alice-2")
        await _until(lambda: session.writes["reason"] == "session_held")
        keys.remove("alice-2")
        await _until(lambda: session.writes == ON)

        assert not waiter.done(), "a session coming or going is not a move"
        await _cancelled(waiter)
        await session.close()

    @pytest.mark.usefixtures("holds")
    async def test_a_provider_that_fails_refuses_the_writes(self, armed_va):
        write_record(armed_va)

        def broken() -> list[str]:
            raise OSError("unreadable")

        session = await ControlSession.open(surface="panel", session_keys=broken)

        assert session.writes == {**_off("record_unavailable"), "kind": None}
        await session.close()
