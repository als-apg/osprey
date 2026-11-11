"""Tests for the gated history tools, in process on fake backends.

``alarm_state``, ``alarm_config``, ``alarm_history`` and ``put_history`` register
only when their predicate holds, so the vocabulary guard over the server's
module-level tools never sees them: this file registers them on a throwaway
server and runs the guard itself. The facts reach backends only through the
directory service, which builds every backend through the factory's create
API; each test swaps those calls for fakes.
"""

from __future__ import annotations

import asyncio
import json
import sys
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

contract = pytest.importorskip("osprey_connectors.pv_directory.contract")

from fastmcp import FastMCP  # noqa: E402

import osprey.channel_roster as channel_roster  # noqa: E402
from osprey.mcp_server.pv_directory import server  # noqa: E402
from osprey.mcp_server.pv_directory.tools import (  # noqa: E402
    alarm_config,
    alarm_history,
    alarm_state,
    put_history,
)
from osprey.services.pv_directory import _connectors  # noqa: E402
from osprey.services.pv_directory.facts import panel_link  # noqa: E402
from osprey_connectors import control_context  # noqa: E402
from osprey_connectors.factory import ConnectorFactory  # noqa: E402
from tests._vocabulary import PROTOCOL_WORDS  # noqa: E402
from tests.mcp_server.conftest import assert_raises_error  # noqa: E402

_REAL_ALARM_STATE = ConnectorFactory.create_alarm_state_connector

VA_BUILD = {"type": "virtual_accelerator"}
LIVE_BUILD = {"type": "epics"}
FACILITY = {"type": "channelfinder", "settings": {"url": "http://cf.example", "timeout_s": 0.5}}
ALARM_LOG = {"type": "phoebus_alarm_logger", "settings": {"url": "http://al.example"}}
PUT_LOG = {"type": "caputlog", "settings": {"url": "http://pl.example"}}
KAFKA = {
    "type": "phoebus_alarm_server",
    "settings": {"bootstrap_servers": "kafka:9092", "topic": "Accelerator"},
}
EVERY_SERVICE = {**FACILITY, "alarm_log": ALARM_LOG, "alarm_state": KAFKA, "put_log": PUT_LOG}
TOOLS = {"alarm_state", "alarm_config", "alarm_history", "put_history"}


# -- fakes --------------------------------------------------------------------


class FakeBackend:
    label = "All PVs"
    empty_reason = "no logged events"

    def __init__(self, spec: Any, *, project_dir: Path | None = None, **seams: Any) -> None:  # noqa: ARG002
        self.spec = spec

    @property
    def source(self) -> Any:
        return contract.Source(
            target=self.spec.target,
            service=self.spec.type,
            label=self.label if self.spec.kind == "pv_directory" else self.spec.type,
            rehearsal=self.spec.rehearsal,
        )

    def _page(self, rows: list[Any]) -> Any:
        if not rows:
            return contract.HistoryPage(
                outcome=contract.Outcome.NO_MATCH,
                source=self.source,
                reason=self.empty_reason,
            )
        return contract.HistoryPage(
            outcome=contract.Outcome.FOUND,
            source=self.source,
            rows=rows,
        )


class FakeDirectory(FakeBackend):
    async def count(self, query: Any, deadline: float) -> Any:  # noqa: ARG002 - a fake
        return 0

    async def describe(self, name: str, deadline: float) -> Any:  # noqa: ARG002 - a fake
        return contract.Outcome.NOT_LISTED


class FakeAlarmState(FakeBackend):
    """A caught-up feed; ``states`` maps a PV to its answer, ``config`` answers every PV."""

    states: dict[str, Any] = {}
    config: Any = None

    async def start(self) -> None:
        pass

    async def aclose(self) -> None:
        pass

    async def state_for(self, pv: str, deadline: float) -> Any:  # noqa: ARG002 - a fake
        return self.states.get(pv)

    async def config_for(self, pv: str, deadline: float) -> Any:  # noqa: ARG002 - a fake
        return self.config


class FakeAlarmLog(FakeBackend):
    rows: list[Any] = []
    config: Any = None
    asked: list[tuple[str, Any, int]] = []

    async def history(self, pv: str, since: Any, limit: int, deadline: float) -> Any:  # noqa: ARG002
        self.asked.append((pv, since, limit))
        return self._page(self.rows)

    async def config_for(self, pv: str, deadline: float) -> Any:  # noqa: ARG002 - a fake
        return self.config


class FakePutLog(FakeBackend):
    empty_reason = "no logged puts"
    rows: list[Any] = []
    asked: list[tuple[str, Any, int]] = []

    async def history(self, pv: str, since: Any, limit: int, deadline: float) -> Any:  # noqa: ARG002
        self.asked.append((pv, since, limit))
        return self._page(self.rows)


_FAKES = {
    "pv_directory": FakeDirectory,
    "alarm_state": FakeAlarmState,
    "alarm_log": FakeAlarmLog,
    "put_log": FakePutLog,
}


@pytest.fixture(autouse=True)
def fakes(monkeypatch: pytest.MonkeyPatch) -> None:
    def create(kind: str):
        async def create_connector(spec: Any, *, project_dir: Path | None = None, **seams: Any):
            return _FAKES[kind](spec, project_dir=project_dir, **seams)

        return create_connector

    for kind in _FAKES:
        monkeypatch.setattr(ConnectorFactory, f"create_{kind}_connector", create(kind))
    monkeypatch.setattr(FakeAlarmState, "states", {})
    monkeypatch.setattr(FakeAlarmState, "config", None)
    monkeypatch.setattr(FakeAlarmLog, "rows", [])
    monkeypatch.setattr(FakeAlarmLog, "config", None)
    monkeypatch.setattr(FakeAlarmLog, "asked", [])
    monkeypatch.setattr(FakePutLog, "rows", [])
    monkeypatch.setattr(FakePutLog, "asked", [])


class _Record:
    """Where the stubbed control-context record points; ``None`` is no record."""

    def __init__(self) -> None:
        self.value: Any = None

    def move(self, target: str, generation: int) -> None:
        self.value = control_context.ControlContext(target=target, generation=generation)


@pytest.fixture(autouse=True)
def record(monkeypatch: pytest.MonkeyPatch) -> _Record:
    holder = _Record()
    monkeypatch.setattr(control_context, "read_record", lambda **_: holder.value)
    return holder


@pytest.fixture(autouse=True)
def cold_roster() -> Iterator[None]:
    channel_roster._roster_cache.clear()
    yield
    channel_roster._roster_cache.clear()


@pytest.fixture(autouse=True)
def no_context() -> Iterator[None]:
    yield
    server.set_tool_context(None)


def _context(
    tmp_path: Path, control_system: dict[str, Any], section: dict[str, Any] | None
) -> server.ToolContext:
    (tmp_path / "facility.json").write_text(json.dumps({"channels": []}), encoding="utf-8")
    config: dict[str, Any] = {"config_dir": str(tmp_path), "control_system": control_system}
    if control_system == VA_BUILD:
        config["services"] = {"virtual_accelerator": {"port": 5064}}
    if section is not None:
        config["pv_directory"] = section
    return server.build_tool_context(config, project_dir=tmp_path)


def _install(
    tmp_path: Path, control_system: dict[str, Any], section: dict[str, Any] | None
) -> server.ToolContext:
    context = _context(tmp_path, control_system, section)
    server.set_tool_context(context)
    return context


def _registered(context: server.ToolContext) -> dict[str, Any]:
    throwaway = FastMCP("probe")
    server.register_tools(throwaway, context)
    return {tool.name: tool for tool in asyncio.run(throwaway.list_tools())}


def _schema_texts(node: Any) -> Iterator[str]:
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "description" and isinstance(value, str):
                yield value
            else:
                yield from _schema_texts(value)
    elif isinstance(node, list):
        for value in node:
            yield from _schema_texts(value)


# -- registration -------------------------------------------------------------


def test_the_four_tools_name_no_protocol_word_and_register_read_only(tmp_path: Path) -> None:
    tools = _registered(_context(tmp_path, LIVE_BUILD, EVERY_SERVICE))

    assert set(tools) == TOOLS
    for name, tool in tools.items():
        texts = [tool.description or "", *_schema_texts(tool.parameters)]
        assert [PROTOCOL_WORDS.findall(text) for text in texts if PROTOCOL_WORDS.search(text)] == []
        assert tool.title, name
        assert tool.annotations.readOnlyHint is True, name


@pytest.mark.parametrize(
    ("section", "expected"),
    [
        (FACILITY, set()),
        ({**FACILITY, "alarm_log": ALARM_LOG}, {"alarm_history", "alarm_config"}),
        ({**FACILITY, "alarm_state": KAFKA}, {"alarm_state", "alarm_config"}),
        ({"alarm_state": {"replay_file": "demo:example-alarms"}}, {"alarm_state", "alarm_config"}),
        ({**FACILITY, "put_log": PUT_LOG}, {"put_history"}),
        (EVERY_SERVICE, TOOLS),
    ],
    ids=["directory-only", "alarm-log-only", "alarm-state-only", "replay-only", "put-log", "all"],
)
def test_each_tool_registers_only_with_its_sub_block(
    tmp_path: Path, section: dict[str, Any], expected: set[str]
) -> None:
    control_system = VA_BUILD if "type" not in section else LIVE_BUILD
    assert set(_registered(_context(tmp_path, control_system, section))) == expected


# -- alarm_state --------------------------------------------------------------


async def test_alarm_state_answers_each_channel_with_its_panel_link(tmp_path: Path) -> None:
    FakeAlarmState.states = {
        "SR:BPM1": contract.AlarmStateRecord(severity="MAJOR", status="HIHI"),
    }
    _install(tmp_path, LIVE_BUILD, {**FACILITY, "alarm_state": KAFKA})

    result = json.loads(await alarm_state.alarm_state(["ca://SR:BPM1.VAL", "SR:BPM2"]))

    first, second = result["channels"]
    assert first["name"] == "ca://SR:BPM1.VAL"
    assert first["lookup"] == "SR:BPM1"
    assert first["outcome"] == "found"
    assert first["record"]["severity"] == "MAJOR"
    assert first["links"] == panel_link("pv", "SR:BPM1")
    assert first["links"] == {"panel": "/panel/channel-finder/#/pv/SR%3ABPM1"}
    assert second["outcome"] == "not_listed"
    assert second["links"] == panel_link("pv", "SR:BPM2")
    assert result["missing"] == {}


async def test_a_stale_alarm_state_is_reported_as_stale(tmp_path: Path) -> None:
    FakeAlarmState.states = {"SR:BPM1": contract.AlarmStateRecord(severity="MINOR", stale=True)}
    _install(tmp_path, LIVE_BUILD, {**FACILITY, "alarm_state": KAFKA})

    result = json.loads(await alarm_state.alarm_state(["SR:BPM1"]))

    (channel,) = result["channels"]
    assert channel["outcome"] == "found"
    assert channel["record"]["stale"] is True
    assert "stale" in (alarm_state.alarm_state.__doc__ or "")


async def test_a_channel_with_no_answer_is_missing_beside_the_answered(tmp_path: Path) -> None:
    FakeAlarmState.states = {
        "SR:BPM1": contract.AlarmStateRecord(severity="NO_ALARM"),
        "SR:BPM2": contract.Unavailable("alarm broker cannot be reached: refused"),
    }
    _install(tmp_path, LIVE_BUILD, {**FACILITY, "alarm_state": KAFKA})

    result = json.loads(await alarm_state.alarm_state(["SR:BPM1", "SR:BPM2"]))

    assert result["missing"] == {"SR:BPM2": "alarm broker cannot be reached: refused"}
    assert [channel["outcome"] for channel in result["channels"]] == ["found", "unavailable"]


async def test_a_feed_still_loading_is_reported(tmp_path: Path) -> None:
    FakeAlarmState.states = {"SR:BPM1": contract.Unavailable("alarm feed still loading")}
    FakeAlarmState.config = contract.Unavailable("alarm feed still loading")
    _install(tmp_path, LIVE_BUILD, {**FACILITY, "alarm_state": KAFKA})

    with assert_raises_error(error_type="nothing_answered") as refused:
        await alarm_state.alarm_state(["SR:BPM1"])
    with assert_raises_error(error_type="service_unavailable") as config:
        await alarm_config.alarm_config("SR:BPM1")

    assert refused["envelope"]["details"]["missing"] == {"SR:BPM1": "alarm feed still loading"}
    assert config["envelope"]["details"]["reason"] == "alarm feed still loading"
    assert config["envelope"]["details"]["label"] == "live configuration"


async def test_a_missing_kafka_client_is_reported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    kafka = pytest.importorskip("osprey_connectors.alarm_state._phoebus_kafka")
    monkeypatch.setitem(sys.modules, "aiokafka", None)
    monkeypatch.setattr(ConnectorFactory, "create_alarm_state_connector", _REAL_ALARM_STATE)
    context = _install(tmp_path, LIVE_BUILD, {**FACILITY, "alarm_state": KAFKA})

    try:
        with assert_raises_error(error_type="nothing_answered") as refused:
            await alarm_state.alarm_state(["SR:BPM1"])
    finally:
        await context.service.aclose()

    assert refused["envelope"]["details"]["missing"] == {"SR:BPM1": kafka.KAFKA_CLIENT_MISSING}


async def test_alarm_state_on_va_is_answered_from_the_replay(
    tmp_path: Path, record: _Record, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("osprey_connectors.alarm_state._replay")
    monkeypatch.setattr(ConnectorFactory, "create_alarm_state_connector", _REAL_ALARM_STATE)
    record.move("va", 1)
    context = _install(tmp_path, VA_BUILD, {"alarm_state": {"replay_file": "demo:example-alarms"}})
    name = "SR:RF:CAVITY:01:TEMPERATURE:RB"

    try:
        result = json.loads(await alarm_state.alarm_state([name]))
    finally:
        await context.service.aclose()

    (channel,) = result["channels"]
    assert channel["outcome"] == "found"
    assert channel["record"]["severity"] == "MAJOR"
    assert channel["record"]["stale"] is False
    assert channel["source"]["target"] == "va"
    assert channel["links"] == panel_link("pv", name)


async def test_alarm_state_with_no_source_on_the_target_refuses_not_supported(
    tmp_path: Path, record: _Record
) -> None:
    record.move("va", 1)
    _install(tmp_path, VA_BUILD, EVERY_SERVICE)

    with assert_raises_error(error_type="not_supported") as refused:
        await alarm_state.alarm_state(["SR:BPM1", "SR:BPM2"])

    expected = _connectors.unavailable_payload(_connectors.NO_SOURCE_REASON, target="va")
    assert {key: refused["envelope"]["details"][key] for key in expected} == expected


async def test_alarm_state_takes_one_to_fifty_names(tmp_path: Path) -> None:
    _install(tmp_path, LIVE_BUILD, {**FACILITY, "alarm_state": KAFKA})

    with assert_raises_error(error_type="validation_error"):
        await alarm_state.alarm_state([f"SR:PV{i}" for i in range(51)])
    with assert_raises_error(error_type="validation_error"):
        await alarm_state.alarm_state([" "])


# -- alarm_config -------------------------------------------------------------


async def test_alarm_config_from_the_feed_carries_its_label_and_link(tmp_path: Path) -> None:
    FakeAlarmState.config = contract.AlarmConfigRecord(
        in_tree="yes", path="/SR/BPM1", label="live configuration"
    )
    _install(tmp_path, LIVE_BUILD, {**FACILITY, "alarm_state": KAFKA})

    result = json.loads(await alarm_config.alarm_config("SR:BPM1"))

    assert result["outcome"] == "found"
    assert result["label"] == "live configuration"
    assert result["record"]["path"] == "/SR/BPM1"
    assert result["links"] == panel_link("pv", "SR:BPM1")


async def test_alarm_config_with_only_the_logger(tmp_path: Path) -> None:
    FakeAlarmLog.config = contract.AlarmConfigRecord(
        in_tree="no", label="last logged configuration"
    )
    _install(tmp_path, LIVE_BUILD, {**FACILITY, "alarm_log": ALARM_LOG})

    result = json.loads(await alarm_config.alarm_config("ca://SR:BPM1"))

    assert result["name"] == "ca://SR:BPM1"
    assert result["outcome"] == "found"
    assert result["label"] == "last logged configuration"
    assert result["record"]["in_tree"] == "no"
    assert result["source"]["service"] == "phoebus_alarm_logger"
    assert result["links"] == panel_link("pv", "SR:BPM1")


async def test_alarm_state_and_alarm_config_carry_the_configured_caveats(tmp_path: Path) -> None:
    FakeAlarmState.states = {"SR:BPM1": contract.AlarmStateRecord(severity="MINOR")}
    FakeAlarmState.config = contract.AlarmConfigRecord(in_tree="yes", label="live configuration")
    caveats = ["covers the SR tree only"]
    _install(tmp_path, LIVE_BUILD, {**FACILITY, "alarm_state": {**KAFKA, "caveats": caveats}})

    state = json.loads(await alarm_state.alarm_state(["SR:BPM1"]))
    config = json.loads(await alarm_config.alarm_config("SR:BPM1"))

    assert state["channels"][0]["caveats"] == caveats
    assert config["caveats"] == caveats
    assert "caveats" in (alarm_state.alarm_state.__doc__ or "")
    assert "caveats" in (alarm_config.alarm_config.__doc__ or "")


# -- alarm_history and put_history --------------------------------------------


@pytest.mark.parametrize(
    ("tool", "fake", "block"),
    [
        (alarm_history.alarm_history, FakeAlarmLog, "alarm_log"),
        (put_history.put_history, FakePutLog, "put_log"),
    ],
    ids=["alarm_history", "put_history"],
)
async def test_an_empty_history_says_nothing_was_logged_with_its_caveats_and_link(
    tmp_path: Path, tool: Any, fake: Any, block: str
) -> None:
    service = {"alarm_log": ALARM_LOG, "put_log": PUT_LOG}[block]
    caveats = ["the log keeps 30 days"]
    _install(tmp_path, LIVE_BUILD, {**FACILITY, block: {**service, "caveats": caveats}})

    result = json.loads(await tool("SR:BPM1", since="2026-11-01T08:00:00+00:00", limit=5))

    assert result["outcome"] == "no_match"
    assert result["reason"] == fake.empty_reason
    assert result["rows"] == []
    assert result["caveats"] == caveats
    assert result["links"] == panel_link("pv", "SR:BPM1")
    assert fake.asked == [("SR:BPM1", datetime.fromisoformat("2026-11-01T08:00:00+00:00"), 5)]
    assert fake.empty_reason in (tool.__doc__ or "")


async def test_a_history_answers_its_rows_newest_twenty_by_default(tmp_path: Path) -> None:
    FakePutLog.rows = [contract.PutRow(at="t2", pv="SR:BPM1", old="1", new="2", user="op")]
    _install(tmp_path, LIVE_BUILD, {**FACILITY, "put_log": PUT_LOG})

    result = json.loads(await put_history.put_history("SR:BPM1"))

    assert result["outcome"] == "found"
    assert result["rows"][0]["user"] == "op"
    assert FakePutLog.asked == [("SR:BPM1", None, 20)]


async def test_a_naive_since_is_read_in_the_facility_zone(tmp_path: Path) -> None:
    _install(tmp_path, LIVE_BUILD, {**FACILITY, "alarm_log": ALARM_LOG})

    await alarm_history.alarm_history("SR:BPM1", since="2026-11-01 08:00")

    ((_, since, _),) = FakeAlarmLog.asked
    assert since.tzinfo is not None
    assert since.replace(tzinfo=None) == datetime(2026, 11, 1, 8, 0)


async def test_a_history_refuses_a_bad_since_or_limit(tmp_path: Path) -> None:
    _install(tmp_path, LIVE_BUILD, {**FACILITY, "alarm_log": ALARM_LOG})

    with assert_raises_error(error_type="validation_error"):
        await alarm_history.alarm_history("SR:BPM1", since="03.04.2026")
    with assert_raises_error(error_type="validation_error"):
        await alarm_history.alarm_history("SR:BPM1", limit=0)
    assert FakeAlarmLog.asked == []


async def test_a_history_with_no_source_on_the_target_refuses_not_supported(
    tmp_path: Path, record: _Record
) -> None:
    record.move("va", 1)
    _install(tmp_path, VA_BUILD, EVERY_SERVICE)

    for tool in (alarm_history.alarm_history, put_history.put_history, alarm_config.alarm_config):
        with assert_raises_error(error_type="not_supported") as refused:
            await tool("SR:BPM1")
        assert refused["envelope"]["details"]["reason"] == "not available for this target"
