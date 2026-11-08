"""Tests for the PV and IOC facts, over a directory service with fake backends.

The facts reach backends only through :class:`DirectoryService`, so every test
builds a real service and swaps the factory's create calls for fakes, as the
service's own tests do. The control target comes from a stubbed control-context
record, or from the deployment's baseline when there is none.
"""

from __future__ import annotations

import asyncio
import fnmatch
import importlib
import json
import sys
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

contract = pytest.importorskip("osprey_connectors.pv_directory.contract")

import osprey.channel_roster as channel_roster  # noqa: E402
from osprey.services.pv_directory import _connectors, facts, service  # noqa: E402
from osprey_connectors import control_context  # noqa: E402
from osprey_connectors.factory import ConnectorFactory  # noqa: E402

_REAL_ALARM_STATE = ConnectorFactory.create_alarm_state_connector

_NEW_MODULES = (
    "osprey_connectors.pv_directory",
    "osprey_connectors.pv_directory.contract",
    "osprey_connectors.pv_directory.targets",
)

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

LIVE_SOURCE = {"target": "live", "service": "channelfinder", "label": "All PVs", "rehearsal": False}


# -- fakes --------------------------------------------------------------------


class FakeBackend:
    """The construction and source every fake shares, as the contract's base has them."""

    label = "All PVs"

    def __init__(self, spec: Any, *, project_dir: Path | None = None, **seams: Any) -> None:
        self.spec = spec
        self.project_dir = project_dir

    @property
    def source(self) -> Any:
        return contract.Source(
            target=self.spec.target,
            service=self.spec.type,
            label=self.label if self.spec.kind == "pv_directory" else self.spec.type,
            rehearsal=self.spec.rehearsal,
        )

    async def probe(self, deadline: float) -> Any:  # noqa: ARG002 - a fake
        return f"{self.spec.type} answered"


class FakeDirectory(FakeBackend):
    """A directory over fixed records: ``ioc`` and ``status`` exact, ``name`` a glob."""

    records: dict[str, Any] = {}
    fail: str | None = None

    def _hits(self, query: Any) -> list[Any]:
        return [
            record.row
            for record in self.records.values()
            if (not query.ioc or record.row.ioc in query.ioc)
            and (query.status is None or record.row.status == query.status)
            and (query.name is None or fnmatch.fnmatchcase(record.row.name, query.name))
            and query.words in record.row.name
        ]

    async def search(self, query: Any, deadline: float) -> Any:  # noqa: ARG002 - a fake
        if self.fail:
            return contract.DirectoryPage(
                outcome=contract.Outcome.UNAVAILABLE,
                searched="",
                request="",
                source=self.source,
                reason=self.fail,
            )
        hits = self._hits(query)
        start = query.page * query.page_size
        rows = hits[start : start + query.page_size]
        return contract.DirectoryPage(
            outcome=contract.Outcome.FOUND if hits else contract.Outcome.NO_MATCH,
            searched="",
            request="GET /fake",
            source=self.source,
            total=len(hits),
            rows=rows,
            truncated=start + len(rows) < len(hits),
        )

    async def count(self, query: Any, deadline: float) -> Any:  # noqa: ARG002 - a fake
        if self.fail:
            return contract.Unavailable(self.fail)
        return len(self._hits(query))

    async def describe(self, name: str, deadline: float) -> Any:  # noqa: ARG002 - a fake
        if self.fail:
            return contract.Unavailable(self.fail)
        return self.records.get(name, contract.Outcome.NOT_LISTED)


class FakeAlarmState(FakeBackend):
    """A caught-up feed by default; ``state``/``config`` are what it answers."""

    state: Any = None
    config: Any = None

    async def start(self) -> None:
        pass

    async def aclose(self) -> None:
        pass

    async def state_for(self, pv: str, deadline: float) -> Any:  # noqa: ARG002 - a fake
        return self.state

    async def config_for(self, pv: str, deadline: float) -> Any:  # noqa: ARG002 - a fake
        return self.config


class FakeAlarmLog(FakeBackend):
    page: Any = None
    config: Any = None
    delay = 0.0

    async def history(self, pv: str, since: Any, limit: int, deadline: float) -> Any:  # noqa: ARG002
        await asyncio.sleep(self.delay)
        return self.page or contract.HistoryPage(
            outcome=contract.Outcome.FOUND,
            source=self.source,
            rows=[contract.AlarmHistoryRow(time="t1", severity="MAJOR", user="operator one")],
        )

    async def config_for(self, pv: str, deadline: float) -> Any:  # noqa: ARG002 - a fake
        return self.config


class FakePutLog(FakeBackend):
    page: Any = None

    async def history(self, pv: str, since: Any, limit: int, deadline: float) -> Any:  # noqa: ARG002
        return self.page or contract.HistoryPage(
            outcome=contract.Outcome.FOUND,
            source=self.source,
            rows=[contract.PutRow(at="t2", pv=pv, old="1", new="2", user="operator two")],
        )


_FAKES = {
    "pv_directory": FakeDirectory,
    "alarm_state": FakeAlarmState,
    "alarm_log": FakeAlarmLog,
    "put_log": FakePutLog,
}


@pytest.fixture(autouse=True)
def fakes(monkeypatch: pytest.MonkeyPatch) -> None:
    """The create calls return fakes."""

    def create(kind: str):
        async def create_connector(spec: Any, *, project_dir: Path | None = None, **seams: Any):
            return _FAKES[kind](spec, project_dir=project_dir, **seams)

        return create_connector

    for kind in _FAKES:
        monkeypatch.setattr(ConnectorFactory, f"create_{kind}_connector", create(kind))
    monkeypatch.setattr(FakeDirectory, "records", {})
    monkeypatch.setattr(FakeDirectory, "fail", None)
    monkeypatch.setattr(FakeAlarmState, "state", None)
    monkeypatch.setattr(FakeAlarmState, "config", None)
    monkeypatch.setattr(FakeAlarmLog, "page", None)
    monkeypatch.setattr(FakeAlarmLog, "config", None)
    monkeypatch.setattr(FakeAlarmLog, "delay", 0.0)
    monkeypatch.setattr(FakePutLog, "page", None)


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


def _config(
    tmp_path: Path,
    control_system: dict[str, Any],
    section: dict[str, Any] | None,
    *curated: str,
    roster: bool = True,
) -> dict[str, Any]:
    if roster:
        channels = [{"id": address, "role": "none"} for address in curated]
        facility = json.dumps({"channels": channels})
        (tmp_path / "facility.json").write_text(facility, encoding="utf-8")
    config: dict[str, Any] = {"config_dir": str(tmp_path), "control_system": control_system}
    if control_system == VA_BUILD:
        config["services"] = {"virtual_accelerator": {"port": 5064}}
    if section is not None:
        config["pv_directory"] = section
    return config


def _facts(config: dict[str, Any], tmp_path: Path) -> Any:
    return facts.Facts(service.DirectoryService(config, project_dir=tmp_path), config)


def _record(name: str, ioc: str = "ioc-a", **properties: str) -> Any:
    status = properties.pop("status", "Active")
    row = contract.DirectoryRow(name=name, ioc=ioc, host="host-a", status=status)
    return contract.DirectoryRecord(row=row, properties=properties)


def _records(*records: Any) -> dict[str, Any]:
    return {record.row.name: record for record in records}


def _sections(answer: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {section["section"]: section for section in answer["sections"]}


# -- the PV page --------------------------------------------------------------


async def test_every_configured_section_answers(tmp_path: Path) -> None:
    FakeDirectory.records = _records(_record("SR:BPM1", recordDesc="beam position"))
    FakeAlarmState.state = contract.AlarmStateRecord(severity="MINOR", stale=True)
    FakeAlarmState.config = contract.AlarmConfigRecord(
        in_tree="yes", path="/SR/BPM1", label="live configuration"
    )
    page = _facts(_config(tmp_path, LIVE_BUILD, EVERY_SERVICE, "SR:BPM1"), tmp_path)

    answer = await page.pv_facts("SR:BPM1")

    assert answer["name"] == "SR:BPM1"
    sections = _sections(answer)
    assert list(sections) == [
        "directory",
        "alarm_state",
        "alarm_config",
        "alarm_history",
        "put_history",
    ]
    directory = sections["directory"]
    assert directory["outcome"] == "found"
    assert directory["origin"] == "curated"
    assert directory["source"] == LIVE_SOURCE
    assert directory["record"]["properties"] == {"recordDesc": "beam position"}
    assert directory["detail_properties"] == ["recordDesc"]
    assert sections["alarm_state"]["outcome"] == "found"
    assert sections["alarm_state"]["record"]["severity"] == "MINOR"
    assert sections["alarm_state"]["record"]["stale"] is True
    assert sections["alarm_state"]["source"]["service"] == "phoebus_alarm_server"
    assert sections["alarm_config"]["label"] == "live configuration"
    assert sections["alarm_config"]["record"]["in_tree"] == "yes"
    assert sections["alarm_history"]["rows"][0]["user"] == "operator one"
    assert sections["put_history"]["rows"][0]["user"] == "operator two"
    assert "reason" not in sections["put_history"]


async def test_an_unconfigured_section_is_absent(tmp_path: Path) -> None:
    FakeDirectory.records = _records(_record("SR:BPM1"))
    page = _facts(_config(tmp_path, LIVE_BUILD, FACILITY), tmp_path)

    sections = _sections(await page.pv_facts("SR:BPM1"))

    assert list(sections) == ["directory"]


async def test_no_section_answers_no_sections(tmp_path: Path) -> None:
    page = _facts(_config(tmp_path, LIVE_BUILD, None), tmp_path)

    assert await page.pv_facts("SR:BPM1") == {"name": "SR:BPM1", "sections": []}


async def test_hidden_properties_stay_hidden_and_detail_properties_come_first(
    tmp_path: Path,
) -> None:
    FakeDirectory.records = _records(_record("SR:BPM1", recordDesc="d", hostName="h", alias="x"))
    display = {"hidden_properties": ["hostName"], "detail_properties": ["alias", "hostName"]}
    page = _facts(_config(tmp_path, LIVE_BUILD, {**FACILITY, "display": display}), tmp_path)

    directory = _sections(await page.pv_facts("SR:BPM1"))["directory"]

    assert directory["record"]["properties"] == {"recordDesc": "d", "alias": "x"}
    assert directory["detail_properties"] == ["alias"]


async def test_a_curated_pv_the_directory_does_not_list(tmp_path: Path) -> None:
    page = _facts(_config(tmp_path, LIVE_BUILD, FACILITY, "SR:GHOST"), tmp_path)

    directory = _sections(await page.pv_facts("SR:GHOST"))["directory"]

    assert directory["outcome"] == "not_listed"
    assert directory["origin"] == "curated"
    assert directory["record"] is None
    assert directory["detail_properties"] == []


async def test_a_listed_pv_the_roster_lacks(tmp_path: Path) -> None:
    FakeDirectory.records = _records(_record("SR:RAW"))
    page = _facts(_config(tmp_path, LIVE_BUILD, FACILITY, "SR:OTHER"), tmp_path)

    directory = _sections(await page.pv_facts("SR:RAW"))["directory"]

    assert directory["outcome"] == "found"
    assert directory["origin"] == "directory"


async def test_without_a_roster_the_origin_is_unknown(tmp_path: Path) -> None:
    FakeDirectory.records = _records(_record("SR:RAW"))
    page = _facts(_config(tmp_path, LIVE_BUILD, FACILITY, roster=False), tmp_path)

    directory = _sections(await page.pv_facts("SR:RAW"))["directory"]

    assert directory["origin"] == "unknown"


async def test_an_inactive_record(tmp_path: Path) -> None:
    FakeDirectory.records = _records(_record("SR:OLD", status="Inactive"))
    page = _facts(_config(tmp_path, LIVE_BUILD, FACILITY), tmp_path)

    directory = _sections(await page.pv_facts("SR:OLD"))["directory"]

    assert directory["outcome"] == "inactive"


async def test_a_failing_directory_leaves_the_other_sections(tmp_path: Path) -> None:
    FakeDirectory.fail = "login page instead of JSON"
    section = {**FACILITY, "put_log": PUT_LOG}
    page = _facts(_config(tmp_path, LIVE_BUILD, section, "SR:OTHER"), tmp_path)

    sections = _sections(await page.pv_facts("SR:BPM1"))

    assert sections["directory"]["outcome"] == "unavailable"
    assert sections["directory"]["reason"] == "login page instead of JSON"
    assert sections["directory"]["origin"] == "directory"
    assert sections["put_history"]["outcome"] == "found"


async def test_a_slow_history_misses_its_own_deadline_only(tmp_path: Path) -> None:
    FakeAlarmLog.delay = 5.0
    FakeDirectory.records = _records(_record("SR:BPM1"))
    section = {**FACILITY, "alarm_log": ALARM_LOG, "put_log": PUT_LOG}
    page = _facts(_config(tmp_path, LIVE_BUILD, section), tmp_path)

    sections = _sections(await asyncio.wait_for(page.pv_facts("SR:BPM1"), 3.0))

    history = sections["alarm_history"]
    assert history["outcome"] == "unavailable"
    assert history["reason"] == "no answer within 0.5 s"
    assert history["source"]["service"] == "phoebus_alarm_logger"
    assert history["rows"] == []
    assert sections["directory"]["outcome"] == "found"
    assert sections["put_history"]["outcome"] == "found"


async def test_an_unavailable_history_page_keeps_its_reason(tmp_path: Path) -> None:
    section = {**FACILITY, "put_log": PUT_LOG}
    page = _facts(_config(tmp_path, LIVE_BUILD, section), tmp_path)
    FakePutLog.page = contract.HistoryPage(
        outcome=contract.Outcome.UNAVAILABLE,
        source=contract.Source(target="live", service="caputlog", label="x", rehearsal=False),
        reason="malformed response",
    )

    history = _sections(await page.pv_facts("SR:BPM1"))["put_history"]

    assert history["outcome"] == "unavailable"
    assert history["reason"] == "malformed response"


async def test_sections_with_no_source_on_the_target(tmp_path: Path, record: _Record) -> None:
    record.move("va", 1)
    page = _facts(_config(tmp_path, VA_BUILD, EVERY_SERVICE), tmp_path)

    sections = _sections(await page.pv_facts("SR:BPM1"))

    for key in ("alarm_state", "alarm_config", "alarm_history", "put_history"):
        assert sections[key]["outcome"] == "unavailable", key
        assert sections[key]["reason"] == "not available for this target", key
        assert sections[key]["source"] == {
            "target": "va",
            "service": "none",
            "label": "All PVs",
            "rehearsal": False,
        }
    assert sections["alarm_config"]["record"]["in_tree"] == "unknown"


async def test_a_backend_that_cannot_be_built_says_why(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def refuse(*_: Any, **__: Any) -> Any:
        raise RuntimeError("bad url")

    monkeypatch.setattr(ConnectorFactory, "create_alarm_log_connector", refuse)
    section = {**FACILITY, "alarm_log": ALARM_LOG}
    page = _facts(_config(tmp_path, LIVE_BUILD, section), tmp_path)

    history = _sections(await page.pv_facts("SR:BPM1"))["alarm_history"]

    assert history["outcome"] == "unavailable"
    assert history["reason"] == "cannot build the alarm_log backend 'phoebus_alarm_logger': bad url"


# -- alarm state and alarm configuration ---------------------------------------


async def test_a_feed_still_loading_is_unavailable_and_never_falls_back(tmp_path: Path) -> None:
    FakeAlarmState.state = contract.Unavailable("alarm feed still loading")
    FakeAlarmState.config = contract.Unavailable("alarm feed still loading")
    FakeAlarmLog.config = contract.AlarmConfigRecord(
        in_tree="no", label="last logged configuration"
    )
    section = {**FACILITY, "alarm_state": KAFKA, "alarm_log": ALARM_LOG}
    page = _facts(_config(tmp_path, LIVE_BUILD, section), tmp_path)

    sections = _sections(await page.pv_facts("SR:BPM1"))

    assert sections["alarm_state"]["outcome"] == "unavailable"
    assert sections["alarm_state"]["reason"] == "alarm feed still loading"
    config = sections["alarm_config"]
    assert config["outcome"] == "unavailable"
    assert config["reason"] == "alarm feed still loading"
    assert config["label"] == "live configuration"
    assert config["record"]["in_tree"] == "unknown"
    assert "reason" not in config["record"]


async def test_a_silent_feed_falls_back_to_the_logged_configuration(tmp_path: Path) -> None:
    FakeAlarmState.config = contract.Unavailable("no message from the feed yet")
    FakeAlarmLog.config = contract.AlarmConfigRecord(
        in_tree="yes", path="/SR/BPM1", label="last logged configuration"
    )
    section = {**FACILITY, "alarm_state": KAFKA, "alarm_log": ALARM_LOG}
    page = _facts(_config(tmp_path, LIVE_BUILD, section), tmp_path)

    config = _sections(await page.pv_facts("SR:BPM1"))["alarm_config"]

    assert config["outcome"] == "found"
    assert config["label"] == "last logged configuration"
    assert config["record"]["path"] == "/SR/BPM1"
    assert config["source"]["service"] == "phoebus_alarm_logger"


async def test_the_logger_alone_answers_the_configuration(tmp_path: Path) -> None:
    FakeAlarmLog.config = contract.Unavailable("second step failed")
    section = {**FACILITY, "alarm_log": ALARM_LOG}
    page = _facts(_config(tmp_path, LIVE_BUILD, section), tmp_path)

    sections = _sections(await page.pv_facts("SR:BPM1"))

    assert "alarm_state" not in sections
    config = sections["alarm_config"]
    assert config["outcome"] == "unavailable"
    assert config["reason"] == "second step failed"
    assert config["record"]["in_tree"] == "unknown"
    assert config["label"] == "last logged configuration"


async def test_a_pv_the_feed_holds_no_state_for(tmp_path: Path) -> None:
    section = {**FACILITY, "alarm_state": KAFKA}
    page = _facts(_config(tmp_path, LIVE_BUILD, section), tmp_path)

    state = _sections(await page.pv_facts("SR:BPM1"))["alarm_state"]

    assert state["outcome"] == "not_listed"
    assert state["record"] is None


async def test_alarm_state_on_va_from_the_replay(
    tmp_path: Path, record: _Record, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("osprey_connectors.alarm_state._replay")
    monkeypatch.setattr(ConnectorFactory, "create_alarm_state_connector", _REAL_ALARM_STATE)
    record.move("va", 1)
    section = {"alarm_state": {"replay_file": "demo:example-alarms"}}
    directory = service.DirectoryService(_config(tmp_path, VA_BUILD, section), project_dir=tmp_path)
    page = facts.Facts(directory, _config(tmp_path, VA_BUILD, section))

    sections = _sections(await page.pv_facts("SR:RF:CAVITY:01:TEMPERATURE:RB"))
    await directory.aclose()

    state = sections["alarm_state"]
    assert state["outcome"] == "found"
    assert state["record"]["severity"] == "MAJOR"
    assert state["record"]["stale"] is False
    assert state["source"]["target"] == "va"
    config = sections["alarm_config"]
    assert config["label"] == "live configuration"
    assert config["record"]["in_tree"] == "yes"


# -- the IOC page -------------------------------------------------------------


async def test_a_live_ioc_by_suffix(tmp_path: Path) -> None:
    FakeDirectory.records = _records(
        _record("SR:BPM1", caPort="5064", pvaPort="5075", Engineer="A. Engineer"),
        _record("SR:BPM2", status="Inactive"),
        _record("SR:IOC-A:HEARTBEAT"),
        _record("SR:IOC-A:UPTIME"),
        _record("SR:IOC-A:STATUS"),
        _record("SR:IOC-B:HEARTBEAT", ioc="ioc-b"),
    )
    config = _config(tmp_path, LIVE_BUILD, FACILITY, "SR:BPM1", "SR:IOC-A:HEARTBEAT")
    page = _facts(config, tmp_path)

    ioc = await page.ioc_facts("ioc-a")

    assert ioc["outcome"] == "found"
    assert ioc["source"] == LIVE_SOURCE
    assert ioc["host"] == "host-a"
    assert ioc["ports"] == [5064, 5075]
    assert ioc["pv_count"] == 5
    assert ioc["inactive_count"] == 1
    assert ioc["curated_count"] == 2
    assert "curated_reason" not in ioc
    assert ioc["engineer"] == "A. Engineer"
    assert ioc["engineer_source"] == "directory property"
    assert ioc["health_pvs"] == ["SR:IOC-A:HEARTBEAT", "SR:IOC-A:UPTIME"]
    assert "health_pvs_reason" not in ioc
    assert ioc["links"] == {"panel": "/panel/channel-finder/#/ioc/ioc-a"}


async def test_a_live_ioc_with_no_health_rows(tmp_path: Path) -> None:
    FakeDirectory.records = _records(_record("SR:BPM1", caPort="5064"))
    page = _facts(_config(tmp_path, LIVE_BUILD, FACILITY), tmp_path)

    ioc = await page.ioc_facts("ioc-a")

    assert ioc["ports"] == [5064]
    assert ioc["health_pvs"] == []
    assert ioc["health_pvs_reason"] == "no IOC health PVs listed for this IOC"
    assert ioc["engineer"] is None
    assert ioc["engineer_source"] is None


async def test_configured_health_suffixes_and_engineer_property(tmp_path: Path) -> None:
    FakeDirectory.records = _records(
        _record("SR:BPM1", Owner="B. Owner", Engineer="ignored"),
        _record("SR:IOC-A:HEARTBEAT"),
        _record("SR:IOC-A:ALIVE"),
    )
    display = {"engineer_property": "Owner", "ioc_health_suffixes": ["ALIVE"]}
    page = _facts(_config(tmp_path, LIVE_BUILD, {**FACILITY, "display": display}), tmp_path)

    ioc = await page.ioc_facts("ioc-a/1")

    assert ioc["outcome"] == "not_listed"
    assert ioc["links"] == {"panel": "/panel/channel-finder/#/ioc/ioc-a%2F1"}
    ioc = await page.ioc_facts("ioc-a")
    assert ioc["engineer"] == "B. Owner"
    assert ioc["health_pvs"] == ["SR:IOC-A:ALIVE"]
    assert ioc["ports"] is None


async def test_a_hidden_engineer_property_is_not_shown(tmp_path: Path) -> None:
    FakeDirectory.records = _records(_record("SR:BPM1", Engineer="A. Engineer"))
    display = {"hidden_properties": ["Engineer"]}
    page = _facts(_config(tmp_path, LIVE_BUILD, {**FACILITY, "display": display}), tmp_path)

    ioc = await page.ioc_facts("ioc-a")

    assert ioc["engineer"] is None
    assert ioc["engineer_source"] is None


async def test_an_ioc_whose_records_are_all_inactive(tmp_path: Path) -> None:
    FakeDirectory.records = _records(_record("SR:OLD", status="Inactive"))
    page = _facts(_config(tmp_path, LIVE_BUILD, FACILITY), tmp_path)

    ioc = await page.ioc_facts("ioc-a")

    assert ioc["outcome"] == "inactive"
    assert ioc["inactive_count"] == 1


async def test_without_a_roster_the_curated_count_says_why(tmp_path: Path) -> None:
    FakeDirectory.records = _records(_record("SR:BPM1"))
    page = _facts(_config(tmp_path, LIVE_BUILD, FACILITY, roster=False), tmp_path)

    ioc = await page.ioc_facts("ioc-a")

    assert ioc["curated_count"] is None
    assert ioc["curated_reason"]


async def test_above_the_filter_cap_the_curated_count_says_why(tmp_path: Path) -> None:
    FakeDirectory.records = _records(*(_record(f"SR:PV{i}") for i in range(4)))
    section = {**FACILITY, "display": {"filter_cap": 3}}
    page = _facts(_config(tmp_path, LIVE_BUILD, section, "SR:PV1"), tmp_path)

    ioc = await page.ioc_facts("ioc-a")

    assert ioc["pv_count"] == 4
    assert ioc["curated_count"] is None
    assert ioc["curated_reason"] == "too many results to filter (4); narrow the search"


async def test_an_ioc_on_an_unavailable_directory(tmp_path: Path) -> None:
    FakeDirectory.fail = "connection refused"
    page = _facts(_config(tmp_path, LIVE_BUILD, FACILITY), tmp_path)

    ioc = await page.ioc_facts("ioc-a")

    assert ioc["outcome"] == "unavailable"
    assert ioc["reason"] == "connection refused"
    assert ioc["source"] == LIVE_SOURCE
    assert ioc["health_pvs"] == []


async def test_the_va_ioc(tmp_path: Path, record: _Record) -> None:
    record.move("va", 1)
    FakeDirectory.records = _records(
        _record("SR:BPM1:X", ioc="sim-SR"),
        _record("ca:SIM:SR:STATUS", ioc="sim-SR"),
        _record("ca:SIM:SR:HEARTBEAT", ioc="sim-SR"),
        _record("ca:SIM:SR:UPTIME", ioc="sim-SR"),
        _record("ca:SIM:SR:ENGINEER", ioc="sim-SR"),
    )
    page = _facts(_config(tmp_path, VA_BUILD, {}), tmp_path)

    ioc = await page.ioc_facts("sim-SR")

    assert ioc["source"]["target"] == "va"
    assert ioc["ports"] is None
    assert ioc["pv_count"] == 5
    assert ioc["engineer_source"] == "ENGINEER PV"
    assert ioc["engineer"] == "ca:SIM:SR:ENGINEER"
    assert ioc["health_pvs"] == ["ca:SIM:SR:ENGINEER", "ca:SIM:SR:HEARTBEAT", "ca:SIM:SR:UPTIME"]


# -- connectors without the PV directory modules -------------------------------


@pytest.fixture
def floor() -> Iterator[tuple[ModuleType, ModuleType]]:
    """The service and facts re-imported over a shim whose new modules are unimportable."""
    with pytest.MonkeyPatch.context() as patch:
        for name in _NEW_MODULES:
            patch.setitem(sys.modules, name, None)
        importlib.reload(_connectors)
        yield importlib.reload(service), importlib.reload(facts)
    importlib.reload(_connectors)
    importlib.reload(service)
    importlib.reload(facts)


async def test_modules_blocked_every_section_answers_the_floor(
    tmp_path: Path, floor: tuple[ModuleType, ModuleType]
) -> None:
    floor_service, floor_facts = floor
    config = _config(tmp_path, LIVE_BUILD, EVERY_SERVICE)
    page = floor_facts.Facts(floor_service.DirectoryService(config, project_dir=tmp_path), config)
    expected = _connectors.unavailable_payload(_connectors.FLOOR_REASON, target="live")

    sections = _sections(await page.pv_facts("SR:BPM1"))
    ioc = await page.ioc_facts("ioc-a")

    assert len(sections) == 5
    for key, section in sections.items():
        assert {field: section[field] for field in expected} == expected, key
    assert {field: ioc[field] for field in expected} == expected
    assert ioc["health_pvs"] == []
