"""CANOPY's PV backend writing through a real control session.

Every test opens a real :class:`~osprey.runtime.session.ControlSession` against
a real control-context record in ``tmp_path``, with the backend built the way
CANOPY builds it. The adapter's own row (2) and its mapping of every session
outcome to CANOPY's exception are checked against OSPREY's write-outcome table
(``fixtures/backend-statement.json``). Each write is judged on the execution
verdict read at its admission: ``confirmed`` at ``asked``, and the hold of every
session the terminal names in ``sessions.json``. The move watcher, ``close()``
and the stale exit are checked with writes in flight.
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import json
import logging
import os
from pathlib import Path
from typing import Any, ClassVar

import pytest
from canopy.contracts import load_write_outcome_vectors
from canopy.epics.config import EpicsConfig
from canopy.epics.gateway import build_connector
from tests._audit_capture import capture_audit_bodies

from osprey.integrations.canopy import SESSIONS_FILENAME
from osprey.integrations.canopy import backend as backend_module
from osprey.integrations.canopy.backend import (
    CLIENT_LABEL,
    SURFACE,
    WRITE_BUDGET_ENV,
    WRITE_DETAIL_MAX_CHARS,
    PVSwitchInProgress,
    PVTargetChanged,
    PVWriteMismatch,
    PVWriteRefused,
    PVWriteUnconfirmed,
    PVWriteUnknown,
    process_write_budget,
)
from osprey.runtime import raw_put_block
from osprey.runtime import session as session_module
from osprey.runtime.budget import WriteBudget
from osprey.runtime.session import SessionWriteResult
from osprey.services.python_executor.execution.wrapper import armed_contract
from osprey_connectors import control_context, posture_store, session_hold, types
from osprey_connectors.control_system import _base as connector_base
from osprey_connectors.control_system._base import ChannelWriteResult, WriteOutcome
from osprey_connectors.control_system._epics_connector import EPICSConnector
from osprey_connectors.control_system._limits_validator import STEP_READ_TIMEOUT_KEY
from osprey_connectors.factory import ConnectorFactory
from osprey_connectors.identity import AUDIT_IDENTITY_ENV, TERMINAL_USER_ENV
from osprey_connectors.refusals import refusal_phrase
from osprey_connectors_tests._fake_connector import FakeConnector
from osprey_connectors_tests._harness import isolated_connector_registries
from osprey_connectors_tests._harness._control_context_fixtures import _set_switch_to

BACKEND_PATH = "osprey.integrations.canopy.backend:OspreyPVConnector"

#: The virtual accelerator with the write tier allowed: with its switch on (the
#: record :func:`write_record` writes) the write verdict allows.
VA_SECTION = {
    "type": types.VIRTUAL_ACCELERATOR,
    "writes": "allowed",
    "connector": {types.VIRTUAL_ACCELERATOR: {"response_delay_ms": 0}},
}

CHANNEL = "SR:QF:01:SP"
ORIGIN = {"screen": "sr/overview", "rev": "r7"}
BINDING = "va@3"

#: The vector rows a backend answers (the rest are FRONT's or the client's).
SERVER_SOURCES = {"adapter", "session", "connector"}

#: The canopy class each outcome maps to.
OUTCOME_CLASSES = {
    "refused": PVWriteRefused,
    "target_changed": PVTargetChanged,
    "switch_in_progress": PVSwitchInProgress,
    "mismatch": PVWriteMismatch,
    "unconfirmed": PVWriteUnconfirmed,
    "unknown": PVWriteUnknown,
}


class _Exited(BaseException):
    """Stands in for the process ending."""

    def __init__(self, status: int) -> None:
        super().__init__(status)
        self.status = status


class _Machine(FakeConnector):
    """The bound connector: records its config, writes and disconnects.

    ``hold`` makes every write wait for it to be set, so a test can keep a write
    in the connector while something else happens.
    """

    configs: ClassVar[list[dict]] = []
    writes: ClassVar[list[tuple[str, Any]]] = []
    connected: ClassVar[int] = 0
    hold: ClassVar[asyncio.Event | None] = None
    stale: ClassVar[asyncio.Event | None] = None

    @classmethod
    def reset(cls) -> None:
        cls.configs, cls.writes, cls.connected = [], [], 0
        cls.hold = cls.stale = None

    async def connect(self, config):
        type(self).configs.append(dict(config))
        await super().connect(config)
        type(self).connected += 1

    async def disconnect(self):
        type(self).connected -= 1
        await super().disconnect()

    def channel_key(self, address):
        """The virtual accelerator's key, which its connector inherits from EPICS."""
        return EPICSConnector.channel_key(self, address)  # type: ignore[arg-type]

    async def write_channel(self, channel_address, value, **kwargs):
        type(self).writes.append((channel_address, value))
        hold = type(self).hold
        if hold is not None:
            await hold.wait()
        return ChannelWriteResult(
            channel_address=channel_address,
            value_written=value,
            outcome=WriteOutcome.CONFIRMED,
            observed_value=value,
        )


class _StaleMachine(_Machine):
    """A connector that can tell when its transport goes stale."""

    async def wait_stale(self):
        await type(self).stale.wait()


def write_record(state_dir: Path, *, target="va", generation=3, switch=None) -> None:
    """Write the record and make its switch exactly *switch* (default: *target* allowed)."""
    path = state_dir / control_context.RECORD_FILENAME
    control_context.write_record(
        control_context.ControlContext(target=target, generation=generation), path=path
    )
    control_context.invalidate_cache()
    if switch is None:
        switch = dict.fromkeys({target} & set(types.CONTROL_TARGETS), "allowed")
    _set_switch_to(path, switch)
    control_context.invalidate_cache()


@pytest.fixture
def audit_dir(monkeypatch):
    """The CANOPY audit bodies this test stores in the agent record, in order."""
    return capture_audit_bodies(monkeypatch, SURFACE)


@pytest.fixture
def armed(monkeypatch):
    """Every arming of the raw-put block, recorded instead of patching this process."""
    calls: list[str] = []
    monkeypatch.setattr(backend_module, "_arm_raw_put_block", lambda: calls.append("armed"))
    return calls


@pytest.fixture
def exits(monkeypatch):
    statuses: list[int] = []

    def fake_exit(status):
        statuses.append(status)
        raise _Exited(status)

    monkeypatch.setattr(backend_module.os, "_exit", fake_exit)
    return statuses


@pytest.fixture
def state_dir(tmp_path, monkeypatch, audit_dir, armed, exits):  # noqa: ARG001 - set up first
    """A session process's environment, bound to ``va@3``, the switch allowed, no hold store."""
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
        if name.startswith(("OSPREY_CONTROL_TARGET", "EPICS_", "CANOPY_EPICS__", "OSPREY_CANOPY_")):
            monkeypatch.delenv(name)
    for name in (
        posture_store.CONTROL_CONTEXT_TREE_ENV_VAR,
        posture_store.CONTROL_OWNER_ENV_VAR,
        posture_store.LAUNCH_POSTURE_ENV_VAR,
        session_hold.HOLD_DIR_ENV_VAR,
        session_module.SESSION_KEY_ENV_VAR,
        "OSPREY_EXECUTION_MODE",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(session_module, "MOVE_POLL_S", 0.01)
    monkeypatch.setattr(connector_base, "_long_lived_session", False)
    monkeypatch.chdir(tmp_path)

    def get_config_value(path, default=None, _config_path=None):
        return VA_SECTION if path == "control_system" else default

    monkeypatch.setattr("osprey_connectors.config.get_config_value", get_config_value)
    write_record(directory)
    monkeypatch.setenv(backend_module.RENDERED_FOR_ENV, BINDING)
    _Machine.reset()
    yield directory
    _Machine.reset()
    control_context.invalidate_cache()


@pytest.fixture
def machine(state_dir):  # noqa: ARG001 - the environment comes first
    with isolated_connector_registries():
        ConnectorFactory.register_control_system(types.VIRTUAL_ACCELERATOR, _Machine)
        yield _Machine


def make_connector():
    return build_connector(EpicsConfig(backend=BACKEND_PATH))


@pytest.fixture
async def connector(machine):  # noqa: ARG001 - registers the connector
    built = make_connector()
    await built.start()
    yield built
    _release_writes()
    await built.close()


def _release_writes() -> None:
    if _Machine.hold is not None:
        _Machine.hold.set()


def records(audit_dir: list[dict]) -> list[dict]:
    return list(audit_dir)


def decisions(audit_dir: list[dict]) -> list[tuple[str, str]]:
    return [(r["decision"], r["reason"]) for r in records(audit_dir)]


async def _until(condition, timeout: float = 2.0) -> bool:
    loop = asyncio.get_running_loop()
    end = loop.time() + timeout
    while loop.time() < end:
        if condition():
            return True
        await asyncio.sleep(0.01)
    return condition()


async def _quiet(task: asyncio.Task | None) -> None:
    if task is None:
        return
    task.cancel()
    with contextlib.suppress(BaseException):
        await task


# -- the vector file ---------------------------------------------------------


#: OSPREY's write-outcome rows for its backend (see the file's ``about``).
STATEMENT_FILE = Path(__file__).parent / "fixtures" / "backend-statement.json"


def load_vectors() -> dict[str, Any]:
    """The backend's write-outcome table."""
    return json.loads(STATEMENT_FILE.read_text(encoding="utf-8"))["write_outcomes"]


VECTORS = load_vectors()
SERVER_ROWS = {
    row_id: row for row_id, row in VECTORS["rows"].items() if row["source"] in SERVER_SOURCES
}


class TestTheVectorFile:
    def test_the_server_rows_are_the_adapter_session_and_connector_rows(self):
        assert set(SERVER_ROWS) == {
            "2",
            "3",
            "3a",
            "3b",
            "3c",
            "4",
            "4b",
            "5",
            "5a",
            "5b",
            "5c",
            "connector-refused",
            "connector-refused-read-only",
            "connector-limits",
            "confirmed",
            "mismatch",
            "unconfirmed",
            "connector-unknown",
        }

    def test_the_connector_rows_are_canopys(self):
        """Each connector row here holds the row of that id in CANOPY's table, byte for byte."""
        canopy_rows = load_write_outcome_vectors()["rows"]
        ours = {
            row_id: row for row_id, row in VECTORS["rows"].items() if row["source"] == "connector"
        }
        assert ours
        assert {row_id: canopy_rows[row_id] for row_id in ours} == ours

    def test_the_detail_limit_is_canopys(self):
        assert load_write_outcome_vectors()["detail_max_chars"] == WRITE_DETAIL_MAX_CHARS

    @pytest.mark.parametrize(
        ("error", "row_id"),
        [
            (PVWriteRefused, "connector-refused"),
            (PVTargetChanged, "connector-target-changed"),
            (PVSwitchInProgress, "connector-switch-in-progress"),
            (PVWriteMismatch, "mismatch"),
            (PVWriteUnconfirmed, "unconfirmed"),
            (PVWriteUnknown, "connector-unknown"),
        ],
    )
    def test_each_error_class_answers_its_row(self, error, row_id):
        row = load_write_outcome_vectors()["rows"][row_id]
        assert error.status_code == row["status"]
        assert error.error_code.value == row["code"]
        assert error().detail == row["default"]


# -- the signature -----------------------------------------------------------


class TestSignature:
    async def test_put_takes_the_binding_and_origin_hare_hands_it(self, connector):
        assert {"binding", "origin"} <= set(inspect.signature(connector.put).parameters)


# -- the verdict, per write ----------------------------------------------------


def _hold_store(tmp_path: Path, monkeypatch, *keys: str) -> Path:
    """A hold store with an empty record for each of *keys*: none of them held."""
    root = tmp_path / "holds"
    monkeypatch.setenv(session_hold.HOLD_DIR_ENV_VAR, str(root))
    for key in keys:
        session_hold.ensure_session(key, root=root)
    return root


def _raise_hold(root: Path, key: str) -> None:
    assert session_hold.open_flag(
        key,
        flag_id="flag-1",
        source="watchdog:limits",
        reason="a limit was refused",
        prompt_ref="turn-7",
        prompt_excerpt="set the current to 900",
        root=root,
    )


def _name_sessions(tmp_path: Path, keys) -> None:
    """The terminal's sessions file in DATA's working directory, as the sidecar writes it."""
    (tmp_path / SESSIONS_FILENAME).write_text(json.dumps(sorted(keys)), encoding="utf-8")


#: What a held session's write answers: the hold's phrase and who raised it.
HELD_DETAIL = refusal_phrase("session_held", "simulator") + " Raised by: watchdog:limits."


class TestAsked:
    """At ``asked`` the viewer asks in place; the write carries ``confirmed``."""

    @pytest.fixture
    def asked(self, state_dir):
        write_record(state_dir, switch={"va": "asked"})
        return state_dir

    @pytest.mark.usefixtures("asked")
    async def test_an_unconfirmed_write_is_refused_and_reaches_nothing(
        self, connector, machine, audit_dir
    ):
        with pytest.raises(PVWriteRefused) as caught:
            await connector.put(CHANNEL, 1.0, binding=BINDING, origin=ORIGIN)
        row = SERVER_ROWS["3a"]
        assert (caught.value.status_code, caught.value.error_code) == (row["status"], row["code"])
        assert (
            caught.value.detail
            == row["detail"]
            == ("This write needs your confirmation; nothing written.")
        )
        assert machine.writes == []
        assert decisions(audit_dir) == [("refused", "approval_missing")]
        assert " position=asked confirmed=no " in records(audit_dir)[0]["detail"]

    @pytest.mark.usefixtures("asked")
    async def test_a_confirmed_write_is_written(self, connector, machine, audit_dir):
        assert (
            await connector.put(CHANNEL, 2.0, binding=BINDING, origin=ORIGIN, confirmed=True)
            is None
        )
        assert machine.writes == [(CHANNEL, 2.0)]
        assert decisions(audit_dir) == [("allowed", "write_attempt"), ("allowed", "write_landed")]
        assert all(" position=asked confirmed=yes " in r["detail"] for r in records(audit_dir))

    async def test_a_move_to_asked_under_an_open_panel_is_read_at_the_next_write(
        self, state_dir, connector, machine, audit_dir, exits
    ):
        """No restart: the position is judged per write."""
        await connector.put(CHANNEL, 1.0, binding=BINDING)
        write_record(state_dir, switch={"va": "asked"})
        with pytest.raises(PVWriteRefused):
            await connector.put(CHANNEL, 2.0, binding=BINDING)
        await connector.put(CHANNEL, 3.0, binding=BINDING, confirmed=True)
        assert machine.writes == [(CHANNEL, 1.0), (CHANNEL, 3.0)]
        assert ("refused", "approval_missing") in decisions(audit_dir)
        assert exits == []


class TestAllowed:
    @pytest.mark.parametrize("confirmed", [False, True])
    async def test_confirmed_is_not_read(self, connector, machine, audit_dir, confirmed):
        await connector.put(CHANNEL, 1.0, binding=BINDING, confirmed=confirmed)
        assert machine.writes == [(CHANNEL, 1.0)]
        assert decisions(audit_dir) == [("allowed", "write_attempt"), ("allowed", "write_landed")]
        assert all(" position=allowed " in r["detail"] for r in records(audit_dir))


class TestTheTerminalsSessions:
    """A panel write belongs to the sessions ``sessions.json`` names: held while any one is."""

    async def test_a_hold_on_any_named_session_refuses_the_write(
        self, tmp_path, monkeypatch, machine, audit_dir
    ):
        root = _hold_store(tmp_path, monkeypatch, "alice-1", "alice-2")
        _name_sessions(tmp_path, ["alice-1", "alice-2"])
        _raise_hold(root, "alice-2")
        built = make_connector()
        await built.start()
        try:
            with pytest.raises(PVWriteRefused) as caught:
                await built.put(CHANNEL, 1.0, binding=BINDING, origin=ORIGIN, confirmed=True)
        finally:
            await built.close()
        assert caught.value.status_code == 403
        assert caught.value.detail == HELD_DETAIL
        assert machine.writes == []
        assert decisions(audit_dir) == [("refused", "session_held")]
        assert " sessions=alice-1,alice-2 " in records(audit_dir)[0]["detail"]

    async def test_a_hold_raised_under_an_open_panel_is_read_at_the_next_write(
        self, tmp_path, monkeypatch, machine, exits
    ):
        root = _hold_store(tmp_path, monkeypatch, "alice-1")
        _name_sessions(tmp_path, ["alice-1"])
        built = make_connector()
        await built.start()
        try:
            await built.put(CHANNEL, 1.0, binding=BINDING)
            _raise_hold(root, "alice-1")
            with pytest.raises(PVWriteRefused) as caught:
                await built.put(CHANNEL, 2.0, binding=BINDING)
            assert caught.value.detail == HELD_DETAIL
            session_hold.answer_flags(
                "alice-1", ["flag-1"], answer="continue", operator="alice", root=root
            )
            await built.put(CHANNEL, 3.0, binding=BINDING)
        finally:
            await built.close()
        assert machine.writes == [(CHANNEL, 1.0), (CHANNEL, 3.0)]
        assert exits == []

    async def test_a_held_session_the_file_does_not_name_holds_nothing(
        self, tmp_path, monkeypatch, machine
    ):
        """Only the sessions the terminal names hold its panel's writes."""
        root = _hold_store(tmp_path, monkeypatch, "alice-1", "alice-2")
        _raise_hold(root, "alice-2")
        _name_sessions(tmp_path, ["alice-1"])
        built = make_connector()
        await built.start()
        try:
            await built.put(CHANNEL, 1.0, binding=BINDING)
        finally:
            await built.close()
        assert machine.writes == [(CHANNEL, 1.0)]

    async def test_no_sessions_file_is_no_hold(self, tmp_path, monkeypatch, machine, audit_dir):
        """No session, no hold: a held key nobody names does not refuse the write."""
        root = _hold_store(tmp_path, monkeypatch, "alice-1")
        _raise_hold(root, "alice-1")
        assert not (tmp_path / SESSIONS_FILENAME).exists()
        built = make_connector()
        await built.start()
        try:
            await built.put(CHANNEL, 1.0, binding=BINDING)
        finally:
            await built.close()
        assert machine.writes == [(CHANNEL, 1.0)]
        assert " sessions=- " in records(audit_dir)[0]["detail"]

    @pytest.mark.parametrize(
        "text",
        [
            pytest.param('["alice-1"', id="not-json"),
            pytest.param('{"alice-1": true}', id="not-a-list"),
            pytest.param("[1, 2]", id="not-strings"),
        ],
    )
    async def test_a_malformed_sessions_file_refuses_the_write(
        self, tmp_path, monkeypatch, machine, audit_dir, text
    ):
        """Unreadable is never read as naming no session."""
        _hold_store(tmp_path, monkeypatch, "alice-1")
        (tmp_path / SESSIONS_FILENAME).write_text(text, encoding="utf-8")
        built = make_connector()
        await built.start()
        try:
            with pytest.raises(PVWriteRefused) as caught:
                await built.put(CHANNEL, 1.0, binding=BINDING, confirmed=True)
        finally:
            await built.close()
        assert caught.value.detail == refusal_phrase("record_unavailable")
        assert machine.writes == []
        assert decisions(audit_dir) == [("refused", "record_unavailable")]


class TestReadSessionKeys:
    def test_a_missing_file_names_none(self, tmp_path):
        assert backend_module.read_session_keys(tmp_path / SESSIONS_FILENAME) == []

    def test_a_list_of_keys_is_read_as_written(self, tmp_path):
        _name_sessions(tmp_path, ["bob-2", "alice-1"])
        assert backend_module.read_session_keys(tmp_path / SESSIONS_FILENAME) == [
            "alice-1",
            "bob-2",
        ]

    @pytest.mark.parametrize("text", ["", "[", "{}", "null", '"alice-1"', '["a", 1]'])
    def test_anything_else_raises(self, tmp_path, text):
        path = tmp_path / SESSIONS_FILENAME
        path.write_text(text, encoding="utf-8")
        with pytest.raises(ValueError):
            backend_module.read_session_keys(path)


# -- row 2 -------------------------------------------------------------------


class TestRowTwo:
    """A binding that is absent, null or not this process's: refused as a changed target."""

    @pytest.mark.parametrize("binding", ["absent", None, "", "va@2", "va@3 ", "fallback", "VA@3"])
    async def test_a_stale_binding_is_refused(self, connector, machine, audit_dir, binding):
        kwargs = {} if binding == "absent" else {"binding": binding}
        with pytest.raises(PVTargetChanged) as caught:
            await connector.put(CHANNEL, 1.0, origin=ORIGIN, **kwargs)
        row = SERVER_ROWS["2"]
        assert (caught.value.status_code, caught.value.error_code) == (row["status"], row["code"])
        assert caught.value.detail == row["detail"] == refusal_phrase("switching", "simulator")
        assert machine.writes == []
        assert decisions(audit_dir) == [("refused", "switching")]

    @pytest.mark.parametrize("confirmed", [False, True])
    async def test_the_refusal_audits_what_the_client_confirmed(
        self, connector, machine, audit_dir, confirmed
    ):
        with pytest.raises(PVTargetChanged):
            await connector.put(CHANNEL, 1.0, binding="va@2", origin=ORIGIN, confirmed=confirmed)
        (record,) = records(audit_dir)
        assert f" confirmed={'yes' if confirmed else 'no'} " in record["detail"]
        assert machine.writes == []

    async def test_row_two_comes_before_the_verdict(self, state_dir, connector, audit_dir):
        """Writes off and a stale binding: the answer is row 2's, not the switch's."""
        write_record(state_dir, switch={})
        with pytest.raises(PVTargetChanged):
            await connector.put(CHANNEL, 1.0, binding="va@2")
        assert decisions(audit_dir) == [("refused", "switching")]


# -- through the session -------------------------------------------------------


class TestThroughTheSession:
    async def test_a_confirmed_write_returns(self, connector, machine, audit_dir):
        """The confirmed row: no exception; the attempt and the landed write are audited."""
        assert await connector.put(CHANNEL, 2.5, binding=BINDING, origin=ORIGIN) is None
        assert machine.writes == [(CHANNEL, 2.5)]
        assert decisions(audit_dir) == [("allowed", "write_attempt"), ("allowed", "write_landed")]
        assert all("sr/overview" in r["detail"] for r in records(audit_dir))

    async def test_the_session_gets_the_bare_name_and_the_origin(self, connector, monkeypatch):
        """The protocol prefix is the client's spelling; the client's wait/timeout go nowhere."""
        calls = []

        async def write(address, value, **kwargs):
            calls.append((address, value, kwargs))
            return SessionWriteResult("confirmed", "write_landed", "written")

        monkeypatch.setattr(connector._session, "write", write)
        await connector.put(
            f"ca://{CHANNEL}", 3, binding=BINDING, origin=ORIGIN, timeout=0.001, wait=False
        )
        assert calls == [(CHANNEL, 3, {"origin": ORIGIN, "confirmed": False})]

    async def test_the_session_gets_the_clients_confirmed(self, connector, monkeypatch):
        calls = []

        async def write(address, value, **kwargs):  # noqa: ARG001 - the session's signature
            calls.append(kwargs["confirmed"])
            return SessionWriteResult("confirmed", "write_landed", "written")

        monkeypatch.setattr(connector._session, "write", write)
        await connector.put(CHANNEL, 3, binding=BINDING, confirmed=True)
        await connector.put(CHANNEL, 3, binding=BINDING)
        assert calls == [True, False]

    async def test_a_write_after_a_move_is_a_changed_target(self, connector, audit_dir, exits):
        """Writes stopped by the move watcher answer row 3 while it drains."""
        connector._session.stop_writes()
        with pytest.raises(PVTargetChanged) as caught:
            await connector.put(CHANNEL, 1.0, binding=BINDING)
        assert caught.value.detail == SERVER_ROWS["3"]["detail"]
        assert decisions(audit_dir) == [("refused", "switching")]
        assert exits == []

    async def test_a_busy_channel_is_row_5b(self, connector, machine, audit_dir):
        """A second write to the channel while the first is in the connector."""
        machine.hold = asyncio.Event()
        first = asyncio.create_task(connector.put(CHANNEL, 1.0, binding=BINDING))
        assert await _until(lambda: machine.writes != [])
        with pytest.raises(PVWriteRefused) as caught:
            await connector.put(f"{CHANNEL}.VAL", 2.0, binding=BINDING)
        assert caught.value.detail == SERVER_ROWS["5b"]["detail"]
        machine.hold.set()
        await first
        assert ("refused", "write_in_progress") in decisions(audit_dir)

    async def test_a_held_write_names_the_canopy_panel_to_a_switch(self, connector, machine):
        """A target switch refused on a write in flight names it as a CANOPY panel write."""
        from osprey.mcp_server.control_system import target_eligibility, target_state

        machine.hold = asyncio.Event()
        first = asyncio.create_task(connector.put(CHANNEL, 1.0, binding=BINDING))
        assert await _until(lambda: machine.writes != [])
        (path,) = target_state.state_dir().glob(f"{target_state.INFLIGHT_FILE_PREFIX}*")
        marker = json.loads(path.read_text())
        assert marker["surface"] == SURFACE == "canopy"
        assert target_eligibility.busy_client(marker) == CLIENT_LABEL == "a CANOPY panel write"
        machine.hold.set()
        await first


def _message_for(row: dict) -> str:
    """What the session says for *row*: its text, or a sample of OSPREY's own."""
    if not row["dynamic"]:
        return row["detail"]
    if row["outcome"] == "mismatch":
        return "written; the channel holds 4.75"
    return "SR:QF:01:SP is outside its limits [0, 10]"


class TestTheTable:
    """Every server row maps to its status, code and detail (the toast)."""

    @pytest.fixture
    async def answering(self, connector, monkeypatch):
        """The started backend, its session answering whatever a case sets."""
        answer: dict[str, SessionWriteResult] = {}

        async def write(address, value, **kwargs):  # noqa: ARG001 - the session's signature
            return answer["result"]

        monkeypatch.setattr(connector._session, "write", write)

        async def put(result: SessionWriteResult):
            answer["result"] = result
            return await connector.put(CHANNEL, 1.0, binding=BINDING, origin=ORIGIN)

        return put

    @pytest.mark.parametrize("row_id", sorted(SERVER_ROWS))
    async def test_each_server_row(self, row_id, answering, connector, audit_dir):
        row = SERVER_ROWS[row_id]
        if row["source"] == "adapter":
            # Row 2 is decided by the adapter itself, not the session.
            call = connector.put(CHANNEL, 1.0, binding="va@2")
            expected_detail = row["detail"]
        else:
            message = _message_for(row)
            call = answering(SessionWriteResult(row["outcome"], row["audit"]["reason"], message))
            expected_detail = message
        if row["status"] == 200:
            assert await call is None
            return
        with pytest.raises(OUTCOME_CLASSES[row["outcome"]]) as caught:
            await call
        assert caught.value.status_code == row["status"]
        assert caught.value.error_code == row["code"]
        assert caught.value.detail == expected_detail
        if row["source"] == "adapter":
            assert decisions(audit_dir) == [(row["audit"]["decision"], row["audit"]["reason"])]

    @pytest.mark.parametrize(
        "row_id", sorted(r for r, row in SERVER_ROWS.items() if row["dynamic"])
    )
    @pytest.mark.parametrize("message", ["", "   ", "\n"])
    async def test_an_empty_message_takes_the_rows_default(self, row_id, message, answering):
        row = SERVER_ROWS[row_id]
        with pytest.raises(OUTCOME_CLASSES[row["outcome"]]) as caught:
            await answering(SessionWriteResult(row["outcome"], row["audit"]["reason"], message))
        assert caught.value.detail == row["default"]

    async def test_a_message_is_stripped(self, answering):
        with pytest.raises(PVWriteRefused) as caught:
            await answering(SessionWriteResult("refused", "channel_write_blocked", "  no.\n"))
        assert caught.value.detail == "no."

    @pytest.mark.parametrize("case", VECTORS["detail_cut_cases"], ids=lambda case: case["case"])
    async def test_each_cut_case_gives_the_detail_it_names(self, case, answering, monkeypatch):
        """The adapter passes the message whole; the envelope detail is CANOPY's cut of it."""
        if case["limit"] != VECTORS["detail_max_chars"]:
            # Bind the case's limit into the cut every write-outcome class makes.
            cut = backend_module._cut_write_detail
            monkeypatch.setattr(
                backend_module, "_cut_write_detail", lambda text: cut(text, case["limit"])
            )
        with pytest.raises(PVWriteRefused) as caught:
            await answering(SessionWriteResult("refused", "channel_write_blocked", case["text"]))
        assert caught.value.detail == case["expected"]

    async def test_an_outcome_nobody_knows_is_unknown(self, answering):
        with pytest.raises(PVWriteUnknown) as caught:
            await answering(SessionWriteResult("sideways", "write_unconfirmed", ""))
        assert caught.value.detail == SERVER_ROWS["connector-unknown"]["detail"]


# -- start: the raw-put block and the budget -----------------------------------


class TestStart:
    async def test_the_raw_put_block_is_armed(self, connector, armed):  # noqa: ARG002
        assert armed == ["armed"]

    def test_arming_is_the_executors_armed_contract(self, monkeypatch):
        """The kernel's contract: rpc stays allowed."""
        calls = []
        monkeypatch.setattr(raw_put_block, "install", lambda *a, **k: calls.append((a, k)))
        backend_module._arm_raw_put_block()
        assert calls == [(("armed",), armed_contract(refuse_rpc=False))]

    async def test_a_block_that_cannot_be_armed_refuses_the_start(
        self, machine, monkeypatch, exits
    ):
        """Status 78, before any connector is built."""

        def fail():
            raise RuntimeError("cannot patch")

        monkeypatch.setattr(backend_module, "_arm_raw_put_block", fail)
        built = make_connector()
        with pytest.raises(_Exited):
            await built.start()
        assert exits == [78]
        assert machine.configs == []

    async def test_without_the_variable_the_config_budget_applies(self, connector, machine):
        session = connector._session
        assert (session._call_bound_s, session._write_bound_s) == (5.0, 12.0)
        assert machine.configs[-1]["timeout_s"] == 5.0
        assert machine.configs[-1][STEP_READ_TIMEOUT_KEY] == 2.0

    async def test_a_smaller_env_budget_lowers_t_s_and_d(self, machine, monkeypatch):
        """T_b, S_b and D_b all come down to the variable's values."""
        monkeypatch.setenv(WRITE_BUDGET_ENV, "1,0.5,2.5")
        built = make_connector()
        await built.start()
        try:
            session = built._session
            assert session._call_bound_s == 1.0
            assert machine.configs[-1]["timeout_s"] == 1.0
            assert machine.configs[-1][STEP_READ_TIMEOUT_KEY] == 0.5
            assert session._write_bound_s == 2.5
        finally:
            await built.close()

    async def test_a_mixed_budget_takes_each_smaller_part(self, machine, monkeypatch):
        """T from the config, S from the variable, D recomputed (not the variable's)."""
        monkeypatch.setenv(WRITE_BUDGET_ENV, "9,0.5,1")
        built = make_connector()
        await built.start()
        try:
            session = built._session
            assert machine.configs[-1]["timeout_s"] == 5.0
            assert machine.configs[-1][STEP_READ_TIMEOUT_KEY] == 0.5
            assert (session._call_bound_s, session._write_bound_s) == (5.0, 10.5)
        finally:
            await built.close()

    async def test_a_malformed_variable_keeps_the_config_budget(
        self,
        machine,  # noqa: ARG002 - registers the connector
        monkeypatch,
        caplog,
    ):
        monkeypatch.setenv(WRITE_BUDGET_ENV, "1,0.5")
        built = make_connector()
        with caplog.at_level(logging.WARNING):
            await built.start()
        try:
            assert (built._session._call_bound_s, built._session._write_bound_s) == (5.0, 12.0)
            assert WRITE_BUDGET_ENV in caplog.text
        finally:
            await built.close()


CONFIGURED = WriteBudget(T=5.0, S=2.0, D=12.0, grace_s=14.0, relay_timeout_s=15.0)


class TestProcessWriteBudget:
    def test_absent_is_the_configured_budget(self):
        assert process_write_budget(CONFIGURED, {}) is CONFIGURED
        assert process_write_budget(CONFIGURED, {WRITE_BUDGET_ENV: "  "}) is CONFIGURED

    def test_smaller_everywhere(self):
        assert process_write_budget(CONFIGURED, {WRITE_BUDGET_ENV: "2,1,5"}) == WriteBudget(
            T=2.0, S=1.0, D=5.0, grace_s=14.0, relay_timeout_s=15.0
        )

    def test_mixed_recomputes_d_and_keeps_grace_and_relay(self):
        """A larger T in the variable never raises the config's; D is S + 2T."""
        budget = process_write_budget(CONFIGURED, {WRITE_BUDGET_ENV: "8,1.5,99"})
        assert budget == WriteBudget(T=5.0, S=1.5, D=11.5, grace_s=14.0, relay_timeout_s=15.0)

    def test_larger_everywhere_changes_nothing(self):
        budget = process_write_budget(CONFIGURED, {WRITE_BUDGET_ENV: "10,4,24"})
        assert budget == CONFIGURED

    @pytest.mark.parametrize(
        "text", ["1,2", "1,2,3,4", "a,b,c", "1,,3", "0,1,2", "-1,1,1", "nan,1,1", "inf,1,1"]
    )
    def test_malformed_is_logged_and_ignored(self, text, caplog):
        with caplog.at_level(logging.WARNING):
            assert process_write_budget(CONFIGURED, {WRITE_BUDGET_ENV: text}) is CONFIGURED
        assert WRITE_BUDGET_ENV in caplog.text


# -- the move watcher, close() and the stale exit --------------------------------


class TestTheMoveWatcher:
    async def test_a_move_stops_writes_drains_closes_and_exits_75(
        self, connector, machine, state_dir, exits
    ):
        watcher = connector._watcher
        session = connector._session
        write_record(state_dir, generation=4)
        with pytest.raises(_Exited):
            await asyncio.wait_for(watcher, 5.0)
        assert exits == [75]
        assert session._writes_stopped
        assert machine.connected == 0

    async def test_a_write_in_flight_finishes_before_the_exit(
        self, connector, machine, state_dir, exits, audit_dir
    ):
        machine.hold = asyncio.Event()
        put = asyncio.create_task(connector.put(CHANNEL, 1.0, binding=BINDING))
        assert await _until(lambda: machine.writes != [])
        watcher = connector._watcher
        write_record(state_dir, generation=4)
        assert await _until(lambda: connector._session._writes_stopped)
        await asyncio.sleep(0.05)
        assert exits == []
        assert not watcher.done()
        machine.hold.set()
        assert await put is None
        with pytest.raises(_Exited):
            await asyncio.wait_for(watcher, 5.0)
        assert exits == [75]
        assert decisions(audit_dir)[-1] == ("allowed", "write_landed")
        assert machine.connected == 0

    async def test_a_failed_drain_logs_and_still_exits(
        self, machine, state_dir, exits, monkeypatch, caplog
    ):
        """A write past its bound does not hold the exit; the transport is left to it."""
        monkeypatch.setenv(WRITE_BUDGET_ENV, "0.1,0.05,0.25")
        monkeypatch.setattr(session_module, "_MARGIN_S", 0.05)
        monkeypatch.setattr(backend_module, "WRITE_MARGIN_S", 0.05)
        built = make_connector()
        await built.start()
        machine.hold = asyncio.Event()
        put = asyncio.create_task(built.put(CHANNEL, 1.0, binding=BINDING))
        try:
            assert await _until(lambda: machine.writes != [])
            watcher = built._watcher
            with caplog.at_level(logging.ERROR):
                write_record(state_dir, generation=4)
                with pytest.raises(_Exited):
                    await asyncio.wait_for(watcher, 5.0)
            assert exits == [75]
            assert "move_drain_incomplete" in caplog.text
            assert machine.connected == 1
        finally:
            machine.hold.set()
            await asyncio.gather(put, return_exceptions=True)
            await built.close()

    async def test_a_drain_that_raises_still_exits(
        self, connector, state_dir, exits, monkeypatch, caplog
    ):
        async def broken(timeout):  # noqa: ARG001 - the drain signature
            raise RuntimeError("drain broke")

        monkeypatch.setattr(connector._session, "drain", broken)
        watcher = connector._watcher
        with caplog.at_level(logging.ERROR):
            write_record(state_dir, generation=4)
            with pytest.raises(_Exited):
                await asyncio.wait_for(watcher, 5.0)
        assert exits == [75]
        assert "move_drain_incomplete" in caplog.text

    async def test_a_failed_watch_is_a_move(self, machine, exits, monkeypatch):
        async def broken(_self):
            raise OSError("record unreadable")

        monkeypatch.setattr(session_module.ControlSession, "wait_moved", broken)
        built = make_connector()
        await built.start()
        with pytest.raises(_Exited):
            await asyncio.wait_for(built._watcher, 5.0)
        assert exits == [75]
        assert machine.connected == 0


class TestClose:
    async def test_parked_watchers_are_cancelled_without_an_exit(self, machine, state_dir, exits):
        built = make_connector()
        await built.start()
        watcher, stale = built._watcher, built._stale_watcher
        await built.close()
        assert watcher.cancelled() and stale.cancelled()
        write_record(state_dir, generation=4)
        await asyncio.sleep(0.05)
        assert exits == []
        assert machine.connected == 0

    async def test_close_drains_a_write_then_disconnects_with_no_exit(
        self, machine, exits, audit_dir
    ):
        built = make_connector()
        await built.start()
        machine.hold = asyncio.Event()
        put = asyncio.create_task(built.put(CHANNEL, 1.0, binding=BINDING))
        assert await _until(lambda: machine.writes != [])
        closing = asyncio.create_task(built.close())
        await asyncio.sleep(0.05)
        assert not closing.done()
        assert machine.connected == 1
        machine.hold.set()
        assert await put is None
        await asyncio.wait_for(closing, 5.0)
        assert exits == []
        assert machine.connected == 0
        assert decisions(audit_dir)[-1] == ("allowed", "write_landed")

    async def test_a_draining_move_watcher_is_not_cancelled(self, machine, state_dir, exits):
        """close() during the move's drain: the watcher runs on to its exit."""
        built = make_connector()
        await built.start()
        machine.hold = asyncio.Event()
        put = asyncio.create_task(built.put(CHANNEL, 1.0, binding=BINDING))
        assert await _until(lambda: machine.writes != [])
        watcher = built._watcher
        session = built._session
        write_record(state_dir, generation=4)
        assert await _until(lambda: session._writes_stopped)
        assert await _until(lambda: watcher in built._acting)
        closing = asyncio.create_task(built.close())
        await asyncio.sleep(0.05)
        assert not watcher.cancelled()
        machine.hold.set()
        await put
        await asyncio.wait_for(closing, 5.0)
        with pytest.raises(_Exited):
            await asyncio.wait_for(watcher, 5.0)
        assert exits == [75]
        assert built._watcher is None


class TestTheStaleExit:
    @pytest.fixture
    def stale_machine(self, state_dir):  # noqa: ARG002 - the environment comes first
        _Machine.reset()
        with isolated_connector_registries():
            ConnectorFactory.register_control_system(types.VIRTUAL_ACCELERATOR, _StaleMachine)
            yield _StaleMachine

    async def test_exit_76_does_not_wait_for_a_write_in_flight(
        self, stale_machine, exits, audit_dir
    ):
        """The attempt record stands; the outcome is unknown; nothing is torn down."""
        stale_machine.stale = asyncio.Event()
        built = make_connector()
        await built.start()
        stale_machine.hold = asyncio.Event()
        put = asyncio.create_task(built.put(CHANNEL, 1.0, binding=BINDING))
        try:
            assert await _until(lambda: stale_machine.writes != [])
            watcher = built._stale_watcher
            stale_machine.stale.set()
            with pytest.raises(_Exited):
                await asyncio.wait_for(watcher, 5.0)
            assert exits == [76]
            assert not put.done()
            assert decisions(audit_dir) == [("allowed", "write_attempt")]
            assert stale_machine.connected == 1
        finally:
            stale_machine.hold.set()
            await asyncio.gather(put, return_exceptions=True)
            await _quiet(built._watcher)
            built._watcher = None
            await built.close()
