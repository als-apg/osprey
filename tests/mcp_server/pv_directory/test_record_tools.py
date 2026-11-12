"""Tests for the record tools, pv_record and ioc_record, in process on fake backends.

The facts reach backends only through the directory service, which builds every
backend through the factory's create API; each test swaps those calls for fakes.
The control target comes from a stubbed control-context record, or from the
deployment's baseline when there is none.
"""

from __future__ import annotations

import asyncio
import fnmatch
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("osprey_connectors.service_answers")

import osprey.channel_roster as channel_roster
from osprey.mcp_server.errors import MAP_KEYS, MISSING_KEY
from osprey.mcp_server.pv_directory import server
from osprey.mcp_server.pv_directory.tools import ioc_record, pv_record
from osprey.services.pv_directory import _connectors
from osprey_connectors import control_context
from osprey_connectors.directory import (
    SERVED_SIGNAL_PROPERTY,
    DirectoryPage,
    DirectoryRecord,
    DirectoryRow,
)
from osprey_connectors.factory import ConnectorFactory
from osprey_connectors.service_answers import (
    HistoryPage,
    Outcome,
    Source,
    Unavailable,
)
from osprey_connectors.simulation.view import ENGINEER, HEARTBEAT, UPTIME
from tests.mcp_server.conftest import assert_raises_error

VA_BUILD = {"type": "virtual_accelerator"}
LIVE_BUILD = {"type": "epics"}
FACILITY = {"type": "channelfinder", "settings": {"url": "http://cf.example", "timeout_s": 0.5}}
PUT_LOG = {"type": "caputlog", "settings": {"url": "http://pl.example"}}
PANEL = "/panel/channel-finder/#/"
SIGNAL = SERVED_SIGNAL_PROPERTY


# -- fakes --------------------------------------------------------------------


class FakeBackend:
    def __init__(self, spec: Any, *, project_dir: Path | None = None, **seams: Any) -> None:  # noqa: ARG002
        self.spec = spec

    @property
    def source(self) -> Any:
        return Source(
            kind=self.spec.kind,
            target=self.spec.target,
            service=self.spec.type,
            rehearsal=self.spec.rehearsal,
        )


class FakeDirectory(FakeBackend):
    """A directory over fixed records; a name in ``failing`` does not answer."""

    records: dict[str, Any] = {}
    failing: set[str] = set()
    fail: str | None = None

    def _hits(self, query: Any) -> list[Any]:
        return [
            record.row
            for record in self.records.values()
            if (not query.ioc or record.row.ioc in query.ioc)
            and (query.status is None or record.row.status == query.status)
            and (query.name is None or fnmatch.fnmatchcase(record.row.name, query.name))
            and query.words in record.row.name
            and all(record.properties.get(key) == value for key, value in query.properties.items())
        ]

    async def search(self, query: Any, deadline: float) -> Any:  # noqa: ARG002 - a fake
        hits = self._hits(query)
        rows = hits[query.page * query.page_size : (query.page + 1) * query.page_size]
        return DirectoryPage(
            outcome=Outcome.FOUND if hits else Outcome.NO_MATCH,
            searched="",
            request="GET /fake",
            source=self.source,
            total=len(hits),
            rows=rows,
            truncated=False,
        )

    async def count(self, query: Any, deadline: float) -> Any:  # noqa: ARG002 - a fake
        if self.fail:
            return Unavailable(self.fail)
        return len(self._hits(query))

    async def describe(self, name: str, deadline: float) -> Any:  # noqa: ARG002 - a fake
        if self.fail or name in self.failing:
            return Unavailable(self.fail or "connection refused")
        return self.records.get(name, Outcome.NOT_LISTED)


class FakePutLog(FakeBackend):
    failing: set[str] = set()

    async def history(self, pv: str, since: Any, limit: int, deadline: float) -> Any:  # noqa: ARG002
        if pv in self.failing:
            return Unavailable("put log did not answer")
        return HistoryPage(outcome=Outcome.FOUND, source=self.source, rows=[])


_FAKES = {"pv_directory": FakeDirectory, "put_log": FakePutLog}


@pytest.fixture(autouse=True)
def fakes(monkeypatch: pytest.MonkeyPatch) -> None:
    def create(kind: str):
        async def create_connector(spec: Any, *, project_dir: Path | None = None, **seams: Any):
            return _FAKES[kind](spec, project_dir=project_dir, **seams)

        return create_connector

    for kind in _FAKES:
        monkeypatch.setattr(ConnectorFactory, f"create_{kind}_connector", create(kind))
    monkeypatch.setattr(FakeDirectory, "records", {})
    monkeypatch.setattr(FakeDirectory, "failing", set())
    monkeypatch.setattr(FakeDirectory, "fail", None)
    monkeypatch.setattr(FakePutLog, "failing", set())


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


def _install(
    tmp_path: Path,
    control_system: dict[str, Any],
    section: dict[str, Any] | None,
    *curated: str,
    roster: bool = True,
) -> None:
    if roster:
        channels = [{"id": address, "role": "none"} for address in curated]
        facility = json.dumps({"channels": channels})
        (tmp_path / "facility.json").write_text(facility, encoding="utf-8")
    config: dict[str, Any] = {"config_dir": str(tmp_path), "control_system": control_system}
    if control_system == VA_BUILD:
        config["services"] = {"virtual_accelerator": {"port": 5064}}
    if section is not None:
        config["pv_directory"] = section
    server.set_tool_context(server.build_tool_context(config, project_dir=tmp_path))


def _record(name: str, ioc: str = "ioc-a", **properties: str) -> Any:
    status = properties.pop("status", "Active")
    row = DirectoryRow(name=name, ioc=ioc, host="host-a", status=status)
    return DirectoryRecord(row=row, properties=properties)


def _records(*records: Any) -> dict[str, Any]:
    return {record.row.name: record for record in records}


def _by_name(result: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {channel["name"]: channel for channel in result["channels"]}


def _directory(channel: dict[str, Any]) -> dict[str, Any]:
    (directory,) = [s for s in channel["sections"] if s["section"] == "directory"]
    return directory


# -- pv_record ----------------------------------------------------------------


async def test_a_found_channel_carries_its_origin_and_panel_link(tmp_path: Path) -> None:
    FakeDirectory.records = _records(_record("SR:BPM1"))
    _install(tmp_path, LIVE_BUILD, FACILITY, "SR:BPM1")

    result = json.loads(await pv_record.pv_record(["SR:BPM1"]))

    channel = _by_name(result)["SR:BPM1"]
    assert _directory(channel)["outcome"] == "found"
    assert _directory(channel)["origin"] == "curated"
    assert channel["links"] == {"panel": f"{PANEL}pv/SR%3ABPM1"}
    assert result["missing"] == {}


async def test_an_inactive_channel_is_answered(tmp_path: Path) -> None:
    FakeDirectory.records = _records(_record("SR:OLD", status="Inactive"))
    _install(tmp_path, LIVE_BUILD, FACILITY, "SR:OTHER")

    result = json.loads(await pv_record.pv_record(["SR:OLD"]))

    channel = _by_name(result)["SR:OLD"]
    assert _directory(channel)["outcome"] == "inactive"
    assert _directory(channel)["origin"] == "directory"
    assert result["missing"] == {}


async def test_a_channel_the_directory_does_not_list_is_answered(tmp_path: Path) -> None:
    _install(tmp_path, LIVE_BUILD, FACILITY, "SR:CURATED")

    result = json.loads(await pv_record.pv_record(["SR:CURATED"]))

    channel = _by_name(result)["SR:CURATED"]
    assert _directory(channel)["outcome"] == "not_listed"
    assert _directory(channel)["origin"] == "curated"
    assert channel["links"] == {"panel": f"{PANEL}pv/SR%3ACURATED"}
    assert result["missing"] == {}


async def test_names_are_normalised_and_keyed_as_asked(tmp_path: Path) -> None:
    FakeDirectory.records = _records(_record("SR:BPM1"))
    _install(tmp_path, LIVE_BUILD, FACILITY)

    result = json.loads(await pv_record.pv_record(["ca://SR:BPM1.VAL", "ca://SR:BPM1.VAL"]))

    (channel,) = result["channels"]
    assert channel["name"] == "ca://SR:BPM1.VAL"
    assert channel["lookup"] == "SR:BPM1"
    assert _directory(channel)["outcome"] == "found"
    assert channel["links"] == {"panel": f"{PANEL}pv/SR%3ABPM1"}


async def test_a_name_with_every_section_unavailable_is_missing(tmp_path: Path) -> None:
    FakeDirectory.records = _records(_record("SR:BPM1"), _record("SR:BPM2"))
    FakeDirectory.failing = {"SR:BPM2", "SR:BPM3"}
    FakePutLog.failing = {"SR:BPM2"}
    _install(tmp_path, LIVE_BUILD, {**FACILITY, "put_log": PUT_LOG})

    result = json.loads(await pv_record.pv_record(["SR:BPM1", "SR:BPM2", "SR:BPM3"]))

    assert result["missing"] == {"SR:BPM2": "connection refused"}
    channels = _by_name(result)
    assert _directory(channels["SR:BPM2"])["outcome"] == "unavailable"
    assert _directory(channels["SR:BPM3"])["outcome"] == "unavailable"
    assert "SR:BPM3" not in result["missing"]


async def test_each_channel_carries_the_facts_outcome(tmp_path: Path) -> None:
    FakeDirectory.records = _records(_record("SR:BPM1"), _record("SR:OLD", status="Inactive"))
    FakeDirectory.failing = {"SR:DOWN"}
    _install(tmp_path, LIVE_BUILD, FACILITY)

    result = json.loads(await pv_record.pv_record(["SR:BPM1", "SR:OLD", "SR:GHOST", "SR:DOWN"]))

    channels = _by_name(result)
    assert channels["SR:BPM1"]["outcome"] == "found"
    assert channels["SR:OLD"]["outcome"] == "found"
    assert channels["SR:GHOST"]["outcome"] == "unknown"
    assert channels["SR:DOWN"]["outcome"] == "unavailable"
    assert channels["SR:DOWN"]["reason"] == "connection refused"
    assert "reason" not in channels["SR:BPM1"]


async def test_nothing_answered_refuses_with_the_missing_map(tmp_path: Path) -> None:
    FakeDirectory.failing = {"SR:BPM1", "SR:BPM2"}
    _install(tmp_path, LIVE_BUILD, FACILITY)

    with assert_raises_error(error_type="nothing_answered") as refused:
        await pv_record.pv_record(["SR:BPM1", "SR:BPM2"])

    envelope = refused["envelope"]
    assert envelope["details"]["missing"] == {
        "SR:BPM1": "connection refused",
        "SR:BPM2": "connection refused",
    }
    assert envelope["error_message"].startswith("pv_record: none of the 2 requested channels")


async def test_no_source_on_the_target_refuses_not_supported(tmp_path: Path) -> None:
    _install(tmp_path, LIVE_BUILD, {"display": {}})

    with assert_raises_error(error_type="not_supported") as refused:
        await pv_record.pv_record(["SR:BPM1"])

    expected = _connectors.unavailable_payload(_connectors.NO_SOURCE_REASON, target="live")
    assert {key: refused["envelope"]["details"][key] for key in expected} == expected


async def test_more_than_fifty_names_is_refused(tmp_path: Path) -> None:
    _install(tmp_path, LIVE_BUILD, FACILITY)

    with assert_raises_error(error_type="validation_error"):
        await pv_record.pv_record([f"SR:PV{i}" for i in range(51)])


def test_pv_record_and_alarm_state_state_a_missing_map() -> None:
    assert MAP_KEYS["pv_record"] == MISSING_KEY
    assert MAP_KEYS["alarm_state"] == MISSING_KEY


# -- ioc_record ---------------------------------------------------------------


async def test_a_live_ioc_returns_its_facts_and_health_channel_names(tmp_path: Path) -> None:
    FakeDirectory.records = _records(
        _record("SR:BPM1", caPort="5064", Engineer="A. Engineer"),
        _record("SR:IOC-A:HEARTBEAT"),
        _record("SR:IOC-A:UPTIME"),
    )
    _install(tmp_path, LIVE_BUILD, FACILITY, "SR:BPM1")

    facts = json.loads(await ioc_record.ioc_record("ioc-a"))

    assert facts["outcome"] == "found"
    assert facts["ports"] == [5064]
    assert facts["pv_count"] == 3
    assert facts["curated_count"] == 1
    assert facts["engineer"] == "A. Engineer"
    assert facts["engineer_source"] == "directory property"
    assert facts["health_pvs"] == ["SR:IOC-A:HEARTBEAT", "SR:IOC-A:UPTIME"]
    assert facts["links"] == {"panel": f"{PANEL}ioc/ioc-a"}


async def test_an_ioc_whose_curated_count_cannot_be_counted_says_why(tmp_path: Path) -> None:
    FakeDirectory.records = _records(_record("SR:BPM1"))
    _install(tmp_path, LIVE_BUILD, FACILITY, roster=False)

    facts = json.loads(await ioc_record.ioc_record("ioc-a"))

    assert facts["curated_count"] is None
    assert facts["curated_reason"]


async def test_the_va_ioc_names_its_engineer_channel(tmp_path: Path, record: _Record) -> None:
    record.move("va", 1)
    FakeDirectory.records = _records(
        _record("ca:SIM:SR:HEARTBEAT", ioc="sim-SR", **{SIGNAL: HEARTBEAT}),
        _record("ca:SIM:SR:UPTIME", ioc="sim-SR", **{SIGNAL: UPTIME}),
        _record("ca:SIM:SR:ENGINEER", ioc="sim-SR", **{SIGNAL: ENGINEER}),
    )
    _install(tmp_path, VA_BUILD, {})

    facts = json.loads(await ioc_record.ioc_record("sim-SR"))

    assert facts["source"]["target"] == "va"
    assert facts["ports"] is None
    assert facts["engineer_source"] == "ENGINEER PV"
    assert facts["engineer"] == "ca:SIM:SR:ENGINEER"
    assert facts["health_pvs"] == [
        "ca:SIM:SR:ENGINEER",
        "ca:SIM:SR:HEARTBEAT",
        "ca:SIM:SR:UPTIME",
    ]


async def test_an_ioc_name_is_one_encoded_link_segment(tmp_path: Path) -> None:
    _install(tmp_path, LIVE_BUILD, FACILITY)

    facts = json.loads(await ioc_record.ioc_record("ioc-a/1"))

    assert facts["outcome"] == "not_listed"
    assert facts["links"] == {"panel": f"{PANEL}ioc/ioc-a%2F1"}


async def test_an_ioc_on_a_directory_that_does_not_answer_refuses(tmp_path: Path) -> None:
    FakeDirectory.fail = "connection refused"
    _install(tmp_path, LIVE_BUILD, FACILITY)

    with assert_raises_error(error_type="service_unavailable") as refused:
        await ioc_record.ioc_record("ioc-a")

    details = refused["envelope"]["details"]
    assert details["outcome"] == "unavailable"
    assert details["reason"] == "connection refused"


async def test_an_ioc_with_no_source_on_the_target_refuses_not_supported(
    tmp_path: Path,
) -> None:
    _install(tmp_path, LIVE_BUILD, {"display": {}})

    with assert_raises_error(error_type="not_supported") as refused:
        await ioc_record.ioc_record("ioc-a")

    assert refused["envelope"]["details"]["reason"] == _connectors.NO_SOURCE_REASON


# -- registration -------------------------------------------------------------


def test_the_record_tools_register_read_only_with_a_title() -> None:
    tools = {tool.name: tool for tool in asyncio.run(server.mcp.list_tools())}
    for name in ("pv_record", "ioc_record"):
        assert tools[name].title
        assert tools[name].annotations.readOnlyHint is True
    assert "channel_read" in tools["ioc_record"].description
    assert "HEARTBEAT" in tools["ioc_record"].description
