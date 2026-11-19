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
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

pytest.importorskip("osprey_connectors.service_answers")

import osprey.channel_roster as channel_roster
from osprey.services.pv_directory import _connectors, facts, service
from osprey.services.pv_directory.alarm_snapshot import AlarmSnapshot
from osprey_connectors import control_context
from osprey_connectors.alarm_log import AlarmLogEntry
from osprey_connectors.alarm_state import (
    AlarmCondition,
    AlarmConfiguration,
    AlarmEntry,
    AlarmSeverity,
    AlarmStates,
)
from osprey_connectors.archiver import ArchiveCoverage, ArchiveExtent
from osprey_connectors.directory import (
    DirectoryDetail,
    DirectoryEntry,
    DirectoryPage,
)
from osprey_connectors.factory import ConnectorFactory
from osprey_connectors.put_log import PutLogEntry
from osprey_connectors.service_answers import (
    LogPage,
    NotListed,
    Source,
    Unavailable,
)
from osprey_connectors.simulation.view import (
    ENGINEER,
    HEARTBEAT,
    SERVED_SIGNAL_PROPERTY,
    STATUS_SIGNAL,
    UPTIME,
)
from tests._service_factory import patch_create_service

SIGNAL = SERVED_SIGNAL_PROPERTY

_REAL_CREATE = ConnectorFactory.create_service

_NEW_MODULES = (
    "osprey_connectors.service_answers",
    "osprey_connectors.service_targets",
    "osprey_connectors._service_http",
    "osprey_connectors.directory",
    "osprey_connectors.directory._contract",
    "osprey_connectors.alarm_state",
    "osprey_connectors.alarm_state._contract",
    "osprey_connectors.alarm_log",
    "osprey_connectors.alarm_log._contract",
    "osprey_connectors.put_log",
    "osprey_connectors.put_log._contract",
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

    def __init__(self, spec: Any, *, project_dir: Path | None = None, **seams: Any) -> None:
        self.spec = spec
        self.project_dir = project_dir

    @property
    def source(self) -> Any:
        return Source(
            kind=self.spec.kind,
            target=self.spec.target,
            service=self.spec.type,
            rehearsal=self.spec.rehearsal,
        )

    async def probe(self, *, timeout_s: float) -> Any:  # noqa: ARG002 - a fake
        return f"{self.spec.type} answered"


class FakeDirectory(FakeBackend):
    """A directory over fixed records: ``ioc`` and ``status`` exact, ``name`` a glob."""

    records: dict[str, Any] = {}
    fail: str | None = None
    #: The property map a ChannelFinder directory answers with its default description.
    field_properties = {
        "status": "pvStatus",
        "channel_type": "recordType",
        "server": "iocName",
        "host": "hostName",
        "announced_at": "time",
        "recorded_stamp": "time",
        "description": "recordDesc",
    }

    def _hits(self, query: Any) -> list[Any]:
        return [
            record.entry
            for record in self.records.values()
            if (not query.server or record.entry.server in query.server)
            and (query.status is None or record.entry.status == query.status)
            and (
                query.address_pattern is None
                or fnmatch.fnmatchcase(record.entry.address, query.address_pattern)
            )
            and query.words in record.entry.address
            and all(record.properties.get(key) == value for key, value in query.properties.items())
        ]

    async def search(self, query: Any, *, timeout_s: float) -> Any:  # noqa: ARG002 - a fake
        if self.fail:
            return Unavailable(cause="unreachable", reason=self.fail)
        hits = self._hits(query)
        start = query.page * query.page_size
        rows = hits[start : start + query.page_size]
        return DirectoryPage(
            outcome="found" if hits else "no_match",
            searched="",
            request="GET /fake",
            source=self.source,
            total=len(hits),
            rows=rows,
            truncated=start + len(rows) < len(hits),
        )

    async def count(self, query: Any, *, timeout_s: float) -> Any:  # noqa: ARG002 - a fake
        if self.fail:
            return Unavailable(cause="unreachable", reason=self.fail)
        return len(self._hits(query))

    async def describe(self, address: str, *, timeout_s: float) -> Any:  # noqa: ARG002 - a fake
        if self.fail:
            return Unavailable(cause="unreachable", reason=self.fail)
        return self.records.get(address, NotListed(address=address))


class FakeAlarmState(FakeBackend):
    """A caught-up feed by default; ``state``/``config`` are what it answers."""

    state: Any = None
    config: Any = None
    listed: Any = None

    async def start(self) -> None:
        pass

    async def states(self, *, timeout_s: float) -> Any:  # noqa: ARG002 - a fake
        return self.listed

    async def aclose(self) -> None:
        pass

    async def state_for(self, address: str, *, timeout_s: float) -> Any:  # noqa: ARG002 - a fake
        return NotListed(address=address) if self.state is None else self.state

    async def config_for(self, address: str, *, timeout_s: float) -> Any:  # noqa: ARG002 - a fake
        return self.config


class FakeAlarmLog(FakeBackend):
    page: Any = None
    config: Any = None
    delay = 0.0

    async def history(self, address: str, **_: Any) -> Any:  # noqa: ARG002 - a fake
        await asyncio.sleep(self.delay)
        return self.page or LogPage(
            outcome="found",
            source=self.source,
            rows=[
                AlarmLogEntry(
                    event_at=datetime(2026, 11, 11, 6, tzinfo=UTC),
                    severity="MAJOR",
                    user="operator one",
                )
            ],
        )

    async def config_for(self, address: str, *, timeout_s: float) -> Any:  # noqa: ARG002 - a fake
        return self.config


class FakePutLog(FakeBackend):
    page: Any = None

    async def history(self, address: str, **_: Any) -> Any:
        return self.page or LogPage(
            outcome="found",
            source=self.source,
            rows=[
                PutLogEntry(
                    recorded_stamp="t2", address=address, old="1", new="2", user="operator two"
                )
            ],
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
        spec_kind = "directory" if kind == "pv_directory" else kind
        patch_create_service(monkeypatch, spec_kind, create(kind))
    monkeypatch.setattr(FakeDirectory, "records", {})
    monkeypatch.setattr(FakeDirectory, "fail", None)
    monkeypatch.setattr(FakeAlarmState, "state", None)
    monkeypatch.setattr(FakeAlarmState, "config", None)
    monkeypatch.setattr(FakeAlarmState, "listed", None)
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
    row = DirectoryEntry(address=name, server=ioc, host="host-a", status=status)
    return DirectoryDetail(entry=row, properties=properties)


def _records(*records: Any) -> dict[str, Any]:
    return {record.entry.address: record for record in records}


def _sections(answer: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {section["section"]: section for section in answer["sections"]}


# -- the PV page --------------------------------------------------------------


async def test_every_configured_section_answers(tmp_path: Path) -> None:
    FakeDirectory.records = _records(_record("SR:BPM1", recordDesc="beam position"))
    FakeAlarmState.state = AlarmCondition(severity="MINOR", stale=True)
    FakeAlarmState.config = AlarmConfiguration(in_tree="yes", path="/SR/BPM1")
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
    assert answer["outcome"] == "found"
    assert all(section["caveats"] == [] for section in answer["sections"])


async def test_an_unconfigured_section_is_absent(tmp_path: Path) -> None:
    FakeDirectory.records = _records(_record("SR:BPM1"))
    page = _facts(_config(tmp_path, LIVE_BUILD, FACILITY), tmp_path)

    sections = _sections(await page.pv_facts("SR:BPM1"))

    assert list(sections) == ["directory"]


async def test_no_section_answers_no_sections(tmp_path: Path) -> None:
    page = _facts(_config(tmp_path, LIVE_BUILD, None), tmp_path)

    assert await page.pv_facts("SR:BPM1") == {
        "name": "SR:BPM1",
        "outcome": "unavailable",
        "reason": "not available for this target",
        "sections": [],
    }


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


async def test_an_unavailable_history_is_served_as_a_timed_out_one_is(tmp_path: Path) -> None:
    section = {**FACILITY, "put_log": PUT_LOG}
    page = _facts(_config(tmp_path, LIVE_BUILD, section), tmp_path)
    FakePutLog.page = Unavailable(cause="bad_reply", reason="malformed response")

    history = _sections(await page.pv_facts("SR:BPM1"))["put_history"]

    # No ``truncated`` or ``retention_known``: the backend gave no page to say them.
    assert history == {
        "section": "put_history",
        "outcome": "unavailable",
        "source": {
            "target": "live",
            "service": "caputlog",
            "label": "caputlog",
            "rehearsal": False,
        },
        "caveats": [],
        "rows": [],
        "reason": "malformed response",
    }


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

    patch_create_service(monkeypatch, "alarm_log", refuse)
    section = {**FACILITY, "alarm_log": ALARM_LOG}
    page = _facts(_config(tmp_path, LIVE_BUILD, section), tmp_path)

    history = _sections(await page.pv_facts("SR:BPM1"))["alarm_history"]

    assert history["outcome"] == "unavailable"
    assert history["reason"] == "cannot build the alarm_log backend 'phoebus_alarm_logger': bad url"


# -- alarm state and alarm configuration ---------------------------------------


async def test_a_feed_still_loading_is_unavailable_and_never_falls_back(tmp_path: Path) -> None:
    FakeAlarmState.state = Unavailable(cause="not_ready", reason="alarm feed still loading")
    FakeAlarmState.config = Unavailable(cause="not_ready", reason="alarm feed still loading")
    FakeAlarmLog.config = AlarmConfiguration(in_tree="no")
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
    FakeAlarmState.config = Unavailable(cause="unreachable", reason="no message from the feed yet")
    FakeAlarmLog.config = AlarmConfiguration(in_tree="yes", path="/SR/BPM1")
    section = {**FACILITY, "alarm_state": KAFKA, "alarm_log": ALARM_LOG}
    page = _facts(_config(tmp_path, LIVE_BUILD, section), tmp_path)

    config = _sections(await page.pv_facts("SR:BPM1"))["alarm_config"]

    assert config["outcome"] == "found"
    assert config["label"] == "last logged configuration"
    assert config["record"]["path"] == "/SR/BPM1"
    assert config["source"]["service"] == "phoebus_alarm_logger"


async def test_the_logger_alone_answers_the_configuration(tmp_path: Path) -> None:
    FakeAlarmLog.config = Unavailable(cause="unreachable", reason="second step failed")
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
    patch_create_service(monkeypatch, "alarm_state", _REAL_CREATE)
    record.move("va", 1)
    caveats = ["replayed, not live"]
    section = {"alarm_state": {"replay_file": "demo:example-alarms", "caveats": caveats}}
    directory = service.DirectoryService(_config(tmp_path, VA_BUILD, section), project_dir=tmp_path)
    page = facts.Facts(directory, _config(tmp_path, VA_BUILD, section))

    sections = _sections(await page.pv_facts("SR:RF:CAVITY:01:TEMPERATURE:RB"))
    await directory.aclose()

    state = sections["alarm_state"]
    assert state["outcome"] == "found"
    assert state["record"]["severity"] == "MAJOR"
    assert state["record"]["stale"] is False
    assert state["source"]["target"] == "va"
    assert state["caveats"] == caveats
    config = sections["alarm_config"]
    assert config["label"] == "live configuration"
    assert config["record"]["in_tree"] == "yes"
    assert config["caveats"] == caveats


# -- every channel in alarm ------------------------------------------------------


def _row(channel: str, severity: str | None, time: str, **state: Any) -> Any:
    """A feed's row of *channel*, its recorded *severity* word read as the fold reads it."""
    member, _acknowledged = AlarmSeverity.of_recorded(severity)
    record = AlarmCondition(severity=member, event_at=datetime.fromisoformat(time), **state)
    return AlarmEntry(address=channel, path=f"/SR/{channel}", condition=record)


def _listed(*rows: Any, stale: bool = False) -> Any:
    contact = datetime(2026, 11, 11, 8, tzinfo=UTC)
    return AlarmStates(rows=rows, stale=stale, last_contact_at=contact)


_EVERY_SEVERITY = (
    _row("SR:OK", "OK", "2026-11-11T07:59:00.000Z"),
    _row("SR:MINOR:OLD", "MINOR", "2026-11-11T07:00:00.000Z"),
    _row("SR:MINOR:NEW", "MINOR", "2026-11-11T07:30:00.000Z", acknowledged=True),
    _row("SR:UNDEFINED", "UNDEFINED", "2026-11-11T07:10:00.000Z"),
    _row("SR:MAJOR", "MAJOR", "2026-11-11T06:00:00.000Z", latched=True, message="HIHI_ALARM"),
    _row("SR:INVALID", "INVALID", "2026-11-11T07:20:00.000Z"),
    _row("SR:NONE", None, "2026-11-11T07:40:00.000Z"),
    _row("SR:UNKNOWN", "DISCONNECTED", "2026-11-11T05:00:00.000Z"),
)


async def test_active_alarms_serve_the_counts_over_every_channel_and_the_active_rows(
    tmp_path: Path,
) -> None:
    FakeAlarmState.listed = _listed(*_EVERY_SEVERITY)
    page = _facts(_config(tmp_path, LIVE_BUILD, {**FACILITY, "alarm_state": KAFKA}), tmp_path)

    answer = await page.active_alarms()

    assert answer["outcome"] == "found"
    assert answer["source"]["service"] == "phoebus_alarm_server"
    assert answer["counts"] == {
        "INVALID": 1,
        "MAJOR": 1,
        "MINOR": 2,
        "OK": 1,
        "UNDEFINED": 2,
    }
    assert answer["total"] == 8
    assert answer["stale"] is False
    assert answer["last_contact"] == "2026-11-11T08:00:00.000Z"
    assert answer["truncated"] is False
    assert [row["pv"] for row in answer["rows"]] == [
        "SR:UNDEFINED",
        "SR:UNKNOWN",
        "SR:INVALID",
        "SR:MAJOR",
        "SR:MINOR:NEW",
        "SR:MINOR:OLD",
    ]
    assert answer["rows"][3] == {
        "pv": "SR:MAJOR",
        "path": "/SR/SR:MAJOR",
        "severity": "MAJOR",
        "status": None,
        "latched": True,
        "acknowledged": False,
        "message": "HIHI_ALARM",
        "time": "2026-11-11T06:00:00.000Z",
        "stale": False,
    }


@pytest.mark.parametrize(
    ("alarm_state", "feed"),
    [
        (KAFKA, "live"),
        ({"type": "alarm_replay", "settings": {"replay_file": "alarms.jsonl"}}, "replay"),
    ],
)
async def test_active_alarms_say_whether_the_feed_is_live_or_a_replay(
    tmp_path: Path, alarm_state: dict[str, Any], feed: str
) -> None:
    FakeAlarmState.listed = _listed(*_EVERY_SEVERITY)
    config = {**FACILITY, "alarm_state": alarm_state}
    page = _facts(_config(tmp_path, LIVE_BUILD, config), tmp_path)

    answer = await page.active_alarms()

    assert answer["source"]["service"] == alarm_state["type"]
    assert answer["feed"] == feed


async def test_an_unknown_severity_is_served_as_undefined_and_leads_major(
    tmp_path: Path,
) -> None:
    FakeAlarmState.listed = _listed(
        _row("SR:MAJOR", "MAJOR", "2026-11-11T07:00:00.000Z"),
        _row("SR:UNKNOWN", "DISCONNECTED", "2026-11-11T05:00:00.000Z"),
    )
    page = _facts(_config(tmp_path, LIVE_BUILD, {**FACILITY, "alarm_state": KAFKA}), tmp_path)

    answer = await page.active_alarms()

    assert [(row["pv"], row["severity"]) for row in answer["rows"]] == [
        ("SR:UNKNOWN", "UNDEFINED"),
        ("SR:MAJOR", "MAJOR"),
    ]


async def test_active_alarms_filter_by_severity_and_a_limit_keeps_the_counts(
    tmp_path: Path,
) -> None:
    FakeAlarmState.listed = _listed(*_EVERY_SEVERITY)
    page = _facts(_config(tmp_path, LIVE_BUILD, {**FACILITY, "alarm_state": KAFKA}), tmp_path)

    minor = await page.active_alarms(severities=["MINOR"])
    cut = await page.active_alarms(limit=2)
    counts_only = await page.active_alarms(limit=0)

    assert [row["pv"] for row in minor["rows"]] == ["SR:MINOR:NEW", "SR:MINOR:OLD"]
    assert minor["counts"]["MAJOR"] == 1
    assert [row["pv"] for row in cut["rows"]] == ["SR:UNDEFINED", "SR:UNKNOWN"]
    assert cut["truncated"] is True
    assert cut["counts"] == minor["counts"]
    assert counts_only["rows"] == []
    assert counts_only["total"] == 8
    assert counts_only["truncated"] is True


async def test_active_alarms_of_a_feed_that_cannot_answer_keep_its_reason(tmp_path: Path) -> None:
    FakeAlarmState.listed = Unavailable(cause="not_ready", reason="alarm feed still loading")
    page = _facts(_config(tmp_path, LIVE_BUILD, {**FACILITY, "alarm_state": KAFKA}), tmp_path)

    answer = await page.active_alarms()

    assert answer["outcome"] == "unavailable"
    assert answer["reason"] == "alarm feed still loading"
    assert answer["source"]["service"] == "phoebus_alarm_server"
    assert "rows" not in answer


async def test_active_alarms_with_no_source_answer_the_no_source_form(tmp_path: Path) -> None:
    page = _facts(_config(tmp_path, LIVE_BUILD, FACILITY), tmp_path)

    answer = await page.active_alarms()

    assert answer["outcome"] == "unavailable"
    assert answer["reason"] == "not available for this target"
    assert answer["source"]["service"] == "none"
    assert answer["caveats"] == []


async def test_active_alarms_on_va_from_the_replay(
    tmp_path: Path, record: _Record, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("osprey_connectors.alarm_state._replay")
    patch_create_service(monkeypatch, "alarm_state", _REAL_CREATE)
    record.move("va", 1)
    section = {"alarm_state": {"replay_file": "demo:example-alarms"}}
    config = _config(tmp_path, VA_BUILD, section)
    directory = service.DirectoryService(config, project_dir=tmp_path)

    answer = await facts.Facts(directory, config).active_alarms()
    await directory.aclose()

    assert answer["counts"] == {"MAJOR": 1, "MINOR": 1, "OK": 1}
    assert [(row["pv"], row["latched"], row["acknowledged"]) for row in answer["rows"]] == [
        ("SR:RF:CAVITY:01:TEMPERATURE:RB", True, False),
        ("SR:VAC:GAUGE:SR03:PRESSURE:RB", False, True),
    ]


# -- caveats --------------------------------------------------------------------


async def test_each_section_carries_the_caveats_of_its_configured_source(tmp_path: Path) -> None:
    FakeDirectory.records = _records(_record("SR:BPM1"))
    FakeAlarmState.state = AlarmCondition(severity="MINOR")
    FakeAlarmState.config = AlarmConfiguration(in_tree="yes")
    section = {
        **FACILITY,
        "caveats": ["names only, no values"],
        "alarm_state": {**KAFKA, "caveats": ["covers the SR tree only"]},
        "alarm_log": {**ALARM_LOG, "caveats": ["the logger keeps 30 days"]},
        "put_log": PUT_LOG,
    }
    page = _facts(_config(tmp_path, LIVE_BUILD, section), tmp_path)

    sections = _sections(await page.pv_facts("SR:BPM1"))

    assert sections["directory"]["caveats"] == ["names only, no values"]
    assert sections["alarm_state"]["caveats"] == ["covers the SR tree only"]
    assert sections["alarm_config"]["caveats"] == ["covers the SR tree only"]
    assert sections["alarm_history"]["caveats"] == ["the logger keeps 30 days"]
    assert sections["put_history"]["caveats"] == []
    assert (await page.alarm_state("SR:BPM1"))["caveats"] == ["covers the SR tree only"]
    assert (await page.alarm_config("SR:BPM1"))["caveats"] == ["covers the SR tree only"]


async def test_the_logged_configuration_carries_the_logger_caveats(tmp_path: Path) -> None:
    FakeAlarmState.config = Unavailable(cause="unreachable", reason="no message from the feed yet")
    FakeAlarmLog.config = AlarmConfiguration(in_tree="yes")
    section = {
        **FACILITY,
        "alarm_state": {**KAFKA, "caveats": ["covers the SR tree only"]},
        "alarm_log": {**ALARM_LOG, "caveats": ["the logger keeps 30 days"]},
    }
    page = _facts(_config(tmp_path, LIVE_BUILD, section), tmp_path)

    config = _sections(await page.pv_facts("SR:BPM1"))["alarm_config"]

    assert config["label"] == "last logged configuration"
    assert config["caveats"] == ["the logger keeps 30 days"]


async def test_a_section_with_no_source_or_that_raised_carries_no_caveats(
    tmp_path: Path, record: _Record, monkeypatch: pytest.MonkeyPatch
) -> None:
    section = {**EVERY_SERVICE, "caveats": ["d"], "alarm_state": {**KAFKA, "caveats": ["s"]}}
    page = _facts(_config(tmp_path, LIVE_BUILD, section), tmp_path)

    async def refuse() -> Any:
        raise RuntimeError("snapshot broke")

    monkeypatch.setattr(page._service, "snapshot", refuse)
    raised = _sections(await page.pv_facts("SR:BPM1"))["alarm_state"]

    assert raised["outcome"] == "unavailable"
    assert raised["reason"] == "the service failed: RuntimeError"
    assert raised["caveats"] == []

    record.move("va", 1)
    sections = _sections(await _facts(_config(tmp_path, VA_BUILD, section), tmp_path).pv_facts("X"))
    for key in ("alarm_state", "alarm_config", "alarm_history", "put_history"):
        assert sections[key]["caveats"] == [], key


async def test_a_history_that_did_not_answer_carries_its_source_caveats(tmp_path: Path) -> None:
    FakeAlarmLog.delay = 5.0
    section = {**FACILITY, "alarm_log": {**ALARM_LOG, "caveats": ["the logger keeps 30 days"]}}
    page = _facts(_config(tmp_path, LIVE_BUILD, section), tmp_path)

    history = _sections(await asyncio.wait_for(page.pv_facts("SR:BPM1"), 3.0))["alarm_history"]

    assert history["outcome"] == "unavailable"
    assert history["caveats"] == ["the logger keeps 30 days"]


@pytest.mark.parametrize("outcome", ["found", "no_match"])
async def test_an_answered_history_carries_the_caveats_of_its_configured_source(
    tmp_path: Path, outcome: str
) -> None:
    if outcome == "no_match":
        FakePutLog.page = LogPage(
            outcome="no_match",
            source=Source(kind="put_log", target="live", service="caputlog", rehearsal=False),
            note="no logged puts",
        )
    section = {
        **FACILITY,
        "alarm_log": {**ALARM_LOG, "caveats": ["the logger keeps 30 days"]},
        "put_log": {**PUT_LOG, "caveats": ["puts through the gateway only"]},
    }
    page = _facts(_config(tmp_path, LIVE_BUILD, section), tmp_path)

    sections = _sections(await page.pv_facts("SR:BPM1"))

    assert sections["alarm_history"]["caveats"] == ["the logger keeps 30 days"]
    puts = sections["put_history"]
    assert puts["caveats"] == ["puts through the gateway only"]
    assert puts["outcome"] == outcome
    if outcome == "no_match":
        # The page's note is served as its ``reason``, after the fields it always had.
        assert puts["reason"] == "no logged puts"
        assert list(puts) == [
            "section",
            "outcome",
            "source",
            "caveats",
            "rows",
            "truncated",
            "retention_known",
            "reason",
        ]
    else:
        assert "reason" not in puts
    history = await page.history("put_log", "SR:BPM1")
    assert history["caveats"] == ["puts through the gateway only"]


async def test_an_answered_history_names_its_configured_source(tmp_path: Path) -> None:
    FakePutLog.page = LogPage(
        outcome="found",
        source=Source(kind="put_log", target="live", service="caputlog", rehearsal=False),
    )
    page = _facts(_config(tmp_path, LIVE_BUILD, {**FACILITY, "put_log": PUT_LOG}), tmp_path)

    history = _sections(await page.pv_facts("SR:BPM1"))["put_history"]

    assert history["source"]["service"] == "caputlog"
    assert history["source"]["label"] == "caputlog"
    assert set(history) == {
        "section",
        "outcome",
        "source",
        "caveats",
        "rows",
        "truncated",
        "retention_known",
    }


# -- the page outcome -----------------------------------------------------------


@pytest.mark.parametrize("status", ["Active", "Inactive"])
async def test_a_listed_pv_is_found(tmp_path: Path, status: str) -> None:
    FakeDirectory.records = _records(_record("SR:BPM1", status=status))
    page = _facts(_config(tmp_path, LIVE_BUILD, FACILITY), tmp_path)

    answer = await page.pv_facts("SR:BPM1")

    assert answer["outcome"] == "found"
    assert "reason" not in answer


async def test_an_unlisted_pv_with_logged_puts_is_not_listed(tmp_path: Path) -> None:
    page = _facts(_config(tmp_path, LIVE_BUILD, {**FACILITY, "put_log": PUT_LOG}), tmp_path)

    answer = await page.pv_facts("SR:GHOST")

    assert _sections(answer)["directory"]["outcome"] == "not_listed"
    assert answer["outcome"] == "not_listed"


async def test_an_unlisted_pv_nothing_else_found_is_unknown(tmp_path: Path) -> None:
    FakePutLog.page = LogPage(
        outcome="found",
        source=Source(kind="put_log", target="live", service="caputlog", rehearsal=False),
    )
    section = {**FACILITY, "alarm_state": KAFKA, "put_log": PUT_LOG}
    page = _facts(_config(tmp_path, LIVE_BUILD, section), tmp_path)

    answer = await page.pv_facts("SR:GHOST")

    assert answer["outcome"] == "unknown"
    assert "reason" not in answer


async def test_a_failing_directory_makes_the_page_unavailable(tmp_path: Path) -> None:
    FakeDirectory.fail = "login page instead of JSON"
    page = _facts(_config(tmp_path, LIVE_BUILD, {**FACILITY, "put_log": PUT_LOG}), tmp_path)

    answer = await page.pv_facts("SR:BPM1")

    assert answer["outcome"] == "unavailable"
    assert answer["reason"] == "login page instead of JSON"
    assert _sections(answer)["put_history"]["outcome"] == "found"


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
    assert ioc["health_pvs"] == [
        {"name": "SR:IOC-A:HEARTBEAT", "role": "heartbeat"},
        {"name": "SR:IOC-A:UPTIME", "role": "uptime"},
    ]
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
    display = {"engineer_property": "Owner", "ioc_health_suffixes": {"heartbeat": "ALIVE"}}
    page = _facts(_config(tmp_path, LIVE_BUILD, {**FACILITY, "display": display}), tmp_path)

    ioc = await page.ioc_facts("ioc-a/1")

    assert ioc["outcome"] == "not_listed"
    assert ioc["links"] == {"panel": "/panel/channel-finder/#/ioc/ioc-a%2F1"}
    ioc = await page.ioc_facts("ioc-a")
    assert ioc["engineer"] == "B. Owner"
    assert ioc["health_pvs"] == [{"name": "SR:IOC-A:ALIVE", "role": "heartbeat"}]
    assert ioc["ports"] is None


async def test_a_suffix_list_assigns_no_role_and_the_default_roles_apply(tmp_path: Path) -> None:
    FakeDirectory.records = _records(
        _record("SR:BPM1"), _record("SR:IOC-A:HEARTBEAT"), _record("SR:IOC-A:ALIVE")
    )
    display = {"ioc_health_suffixes": ["ALIVE"]}
    page = _facts(_config(tmp_path, LIVE_BUILD, {**FACILITY, "display": display}), tmp_path)

    ioc = await page.ioc_facts("ioc-a")

    assert ioc["health_pvs"] == [{"name": "SR:IOC-A:HEARTBEAT", "role": "heartbeat"}]


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


async def test_ioc_facts_with_no_source_carries_no_count_fields(tmp_path: Path) -> None:
    page = _facts(_config(tmp_path, LIVE_BUILD, {"display": {}}), tmp_path)

    ioc = await page.ioc_facts("ioc-a")

    assert set(ioc) == {
        "ioc",
        "outcome",
        "source",
        "host",
        "ports",
        "pv_count",
        "inactive_count",
        "curated_count",
        "engineer",
        "engineer_source",
        "health_pvs",
        "links",
        "reason",
    }
    assert ioc["outcome"] == "unavailable"
    assert ioc["reason"] == _connectors.NO_SOURCE_REASON


async def test_the_va_ioc(tmp_path: Path, record: _Record) -> None:
    record.move("va", 1)
    FakeDirectory.records = _records(
        _record("SR:BPM1:X", ioc="sim-SR"),
        _record("ca:SIM:SR:STATUS", ioc="sim-SR", **{SIGNAL: STATUS_SIGNAL}),
        _record("ca:SIM:SR:HEARTBEAT", ioc="sim-SR", **{SIGNAL: HEARTBEAT}),
        _record("ca:SIM:SR:UPTIME", ioc="sim-SR", **{SIGNAL: UPTIME}),
        _record("ca:SIM:SR:ENGINEER", ioc="sim-SR", **{SIGNAL: ENGINEER}),
    )
    page = _facts(_config(tmp_path, VA_BUILD, {}), tmp_path)

    ioc = await page.ioc_facts("sim-SR")

    assert ioc["source"]["target"] == "va"
    assert ioc["ports"] is None
    assert ioc["pv_count"] == 5
    assert ioc["engineer_source"] == "ENGINEER PV"
    assert ioc["engineer"] == "ca:SIM:SR:ENGINEER"
    assert ioc["health_pvs"] == [
        {"name": "ca:SIM:SR:ENGINEER", "role": "engineer"},
        {"name": "ca:SIM:SR:HEARTBEAT", "role": "heartbeat"},
        {"name": "ca:SIM:SR:UPTIME", "role": "uptime"},
    ]


async def test_on_va_a_health_name_without_the_mark_is_not_a_health_pv(
    tmp_path: Path, record: _Record
) -> None:
    record.move("va", 1)
    FakeDirectory.records = _records(
        _record("ca:SIM:SR:EXTRA:HEARTBEAT", ioc="sim-SR"),
        _record("ca:SIM:SR:ALIVE_COUNTER", ioc="sim-SR", **{SIGNAL: HEARTBEAT}),
    )
    page = _facts(_config(tmp_path, VA_BUILD, {}), tmp_path)

    ioc = await page.ioc_facts("sim-SR")

    assert ioc["health_pvs"] == [{"name": "ca:SIM:SR:ALIVE_COUNTER", "role": "heartbeat"}]


async def test_on_va_the_engineer_is_the_row_marked_engineer(
    tmp_path: Path, record: _Record
) -> None:
    record.move("va", 1)
    FakeDirectory.records = _records(
        _record("ca:SIM:SR:RESPONSIBLE", ioc="sim-SR", **{SIGNAL: ENGINEER}),
        _record("ca:SIM:SR:HEARTBEAT", ioc="sim-SR", **{SIGNAL: HEARTBEAT}),
    )
    page = _facts(_config(tmp_path, VA_BUILD, {}), tmp_path)

    ioc = await page.ioc_facts("sim-SR")

    assert ioc["engineer"] == "ca:SIM:SR:RESPONSIBLE"
    assert ioc["engineer_source"] == "ENGINEER PV"


# -- the IOC's alarms ------------------------------------------------------------


async def test_an_ioc_with_a_configured_feed_carries_its_active_alarms(tmp_path: Path) -> None:
    FakeDirectory.records = _records(
        _record("SR:BPM1"),
        _record("SR:BPM2"),
        _record("SR:BPM3"),
        _record("SR:OTHER", ioc="ioc-b"),
    )
    FakeAlarmState.listed = _listed(
        _row("SR:BPM1", "MAJOR", "2026-11-11T06:00:00.000Z", status="NO_ALARM", latched=True),
        _row("SR:BPM2", "OK", "2026-11-11T07:00:00.000Z"),
        _row("SR:BPM3", "MINOR", "2026-11-11T07:30:00.000Z", acknowledged=True),
        _row("SR:OTHER", "MAJOR", "2026-11-11T07:40:00.000Z"),
    )
    page = _facts(_config(tmp_path, LIVE_BUILD, {**FACILITY, "alarm_state": KAFKA}), tmp_path)

    alarms = (await page.ioc_facts("ioc-a"))["alarms"]

    assert alarms["outcome"] == "found"
    assert alarms["total"] == 2
    # Per severity over the whole IOC, from the search's alarm facet.
    assert alarms["counts"] == {"MAJOR": 1, "MINOR": 1}
    assert "reason" not in alarms
    assert alarms["rows"] == [
        {
            "pv": "SR:BPM1",
            "severity": "MAJOR",
            "status": "NO_ALARM",
            "latched": True,
            "acknowledged": False,
            "time": "2026-11-11T06:00:00.000Z",
        },
        {
            "pv": "SR:BPM3",
            "severity": "MINOR",
            "status": None,
            "latched": False,
            "acknowledged": True,
            "time": "2026-11-11T07:30:00.000Z",
        },
    ]


async def test_an_ioc_with_nothing_in_alarm_counts_zero(tmp_path: Path) -> None:
    FakeDirectory.records = _records(_record("SR:BPM1"))
    FakeAlarmState.listed = _listed(_row("SR:BPM1", "OK", "2026-11-11T06:00:00.000Z"))
    page = _facts(_config(tmp_path, LIVE_BUILD, {**FACILITY, "alarm_state": KAFKA}), tmp_path)

    alarms = (await page.ioc_facts("ioc-a"))["alarms"]

    assert alarms == {"outcome": "no_match", "total": 0, "rows": [], "counts": {}}


async def test_a_feed_still_loading_leaves_the_ioc_alarms_unavailable_never_zero(
    tmp_path: Path,
) -> None:
    FakeDirectory.records = _records(_record("SR:BPM1"))
    FakeAlarmState.listed = Unavailable(cause="not_ready", reason="alarm feed still loading")
    page = _facts(_config(tmp_path, LIVE_BUILD, {**FACILITY, "alarm_state": KAFKA}), tmp_path)

    alarms = (await page.ioc_facts("ioc-a"))["alarms"]

    assert alarms == {
        "outcome": "unavailable",
        "reason": "cannot filter by alarm: alarm feed still loading",
    }


async def test_an_ioc_on_a_target_with_no_alarm_source_says_so(
    tmp_path: Path, record: _Record
) -> None:
    record.move("va", 1)
    FakeDirectory.records = _records(_record("SR:BPM1:X", ioc="sim-SR"))
    page = _facts(_config(tmp_path, VA_BUILD, {"alarm_state": KAFKA}), tmp_path)

    alarms = (await page.ioc_facts("sim-SR"))["alarms"]

    assert alarms == {
        "outcome": "unavailable",
        "reason": "cannot filter by alarm: not available for this target",
    }


async def test_an_ioc_without_a_configured_feed_carries_no_alarms(tmp_path: Path) -> None:
    FakeDirectory.records = _records(_record("SR:BPM1"))
    page = _facts(_config(tmp_path, LIVE_BUILD, FACILITY), tmp_path)

    assert "alarms" not in await page.ioc_facts("ioc-a")


# -- the archive ----------------------------------------------------------------

APPLIANCE = {"type": "epics_archiver", "settings": {"url": "http://archiver.example"}}


def _covered(archived: bool | None, **fields: Any) -> dict[str, Any]:
    """The fields of an ``ArchiveCoverage`` answer, but its address."""
    return {
        "is_archived": archived,
        "first_sample_at": fields.get("start"),
        "last_sample_at": fields.get("end"),
        "sampling_period_s": fields.get("period"),
        "unknown_reason": fields.get("unknown_reason"),
    }


class FakeArchiver(FakeBackend):
    """An archive answering ``covered`` and ``held``; ``fail`` makes reads raise."""

    covered: Any = None
    held: Any = None
    fail = False

    async def coverage(self, addresses: Any, *, timeout_s: float | None = None) -> Any:  # noqa: ARG002 - a fake
        if self.fail:
            raise RuntimeError("store down")
        return {address: ArchiveCoverage(address=address, **self.covered) for address in addresses}

    async def extent(self, *, timeout_s: float | None = None) -> Any:  # noqa: ARG002 - a fake
        return self.held

    async def aclose(self) -> None:
        pass


@pytest.fixture
def archiver(monkeypatch: pytest.MonkeyPatch) -> type[FakeArchiver]:
    async def create(spec: Any, **kwargs: Any) -> FakeArchiver:
        return FakeArchiver(spec, **kwargs)

    patch_create_service(monkeypatch, "archive", create)
    monkeypatch.setattr(FakeArchiver, "covered", _covered(None, unknown_reason="cannot say"))
    monkeypatch.setattr(
        FakeArchiver,
        "held",
        Unavailable(cause="unsupported", reason="this archiver does not say"),
    )
    monkeypatch.setattr(FakeArchiver, "fail", False)
    return FakeArchiver


def _archived(tmp_path: Path, section: Any = FACILITY, **display: Any) -> dict[str, Any]:
    FakeDirectory.records = _records(
        _record("SR:BPM1"), _record("SR:BPM2"), _record("SR:RF1", ioc="ioc-b")
    )
    config = _config(tmp_path, LIVE_BUILD, {**section, "display": display}, "SR:BPM1")
    config["archiver"] = APPLIANCE
    return config


@pytest.mark.usefixtures("archiver")
async def test_the_archive_section_follows_the_directory_only_with_an_archive(
    tmp_path: Path,
) -> None:
    with_archive = _facts(_archived(tmp_path, {**FACILITY, "put_log": PUT_LOG}), tmp_path)
    without = _facts(_config(tmp_path, LIVE_BUILD, FACILITY), tmp_path)

    keys = [section["section"] for section in (await with_archive.pv_facts("SR:BPM1"))["sections"]]
    plain = [section["section"] for section in (await without.pv_facts("SR:BPM1"))["sections"]]

    assert keys == ["directory", "archive", "put_history"]
    assert plain == ["directory"]


@pytest.mark.usefixtures("archiver")
async def test_an_archiver_that_cannot_say_answers_found_with_archived_null(tmp_path: Path) -> None:
    page = _facts(_archived(tmp_path), tmp_path)

    section = _sections(await page.pv_facts("SR:BPM1"))["archive"]

    assert section["outcome"] == "found"
    assert section["source"] == {
        "target": "live",
        "service": "epics_archiver",
        "label": "epics_archiver",
        "rehearsal": False,
    }
    assert section["caveats"] == []
    assert section["record"] == {
        "archived": None,
        "first_sample": None,
        "last_sample": None,
        "sampling_period_s": None,
        "description": "cannot say",
    }


async def test_an_archived_pv_answers_its_samples_in_utc(
    tmp_path: Path, archiver: type[FakeArchiver]
) -> None:
    archiver.covered = _covered(
        True,
        start=datetime(2026, 11, 1, tzinfo=UTC),
        end=datetime(2026, 11, 12, 8, 0, 4, tzinfo=UTC),
        period=10.0,
    )
    page = _facts(_archived(tmp_path), tmp_path)

    record = _sections(await page.pv_facts("SR:BPM1"))["archive"]["record"]

    assert record["archived"] is True
    assert record["first_sample"] == "2026-11-01T00:00:00+00:00"
    assert record["last_sample"] == "2026-11-12T08:00:04+00:00"
    assert record["sampling_period_s"] == 10.0


async def test_a_raised_archive_read_is_unavailable_with_its_reason(
    tmp_path: Path, archiver: type[FakeArchiver]
) -> None:
    archiver.fail = True
    page = _facts(_archived(tmp_path), tmp_path)

    answer = await page.pv_facts("SR:BPM1")
    section = _sections(answer)["archive"]

    assert section["outcome"] == "unavailable"
    assert section["reason"] == "the service failed: RuntimeError"
    assert answer["outcome"] == "found"


async def test_a_raised_archive_read_never_serves_the_exception_text(
    tmp_path: Path, archiver: type[FakeArchiver], monkeypatch: pytest.MonkeyPatch
) -> None:
    async def leaking(self: FakeArchiver, addresses: Any, **_kwargs: Any) -> Any:  # noqa: ARG001 - a fake
        raise ConnectionError("cannot reach https://reader:secret@archiver.example/mgmt")

    monkeypatch.setattr(archiver, "coverage", leaking)
    page = _facts(_archived(tmp_path), tmp_path)

    section = _sections(await page.pv_facts("SR:BPM1"))["archive"]

    assert section["reason"] == "the service failed: ConnectionError"
    assert "https://" not in str(section) and "secret" not in str(section)


#: A failure whose text holds a login: a served answer names its class only.
LEAK = "cannot reach https://reader:secret@service.example/api"


class _Leak(ConnectionError):
    """A backend failure whose message carries a URL with a login."""


async def _leak(*_args: Any, **_kwargs: Any) -> Any:
    raise _Leak(LEAK)


@pytest.mark.parametrize(
    ("section", "reads"),
    [
        ("directory", [(FakeDirectory, "describe")]),
        ("archive", [(FakeArchiver, "coverage")]),
        ("alarm_state", [(AlarmSnapshot, "state_for")]),
        ("alarm_config", [(AlarmSnapshot, "config_for"), (FakeAlarmLog, "config_for")]),
        ("alarm_history", [(FakeAlarmLog, "history")]),
        ("put_history", [(FakePutLog, "history")]),
    ],
)
@pytest.mark.usefixtures("archiver")
async def test_a_raising_section_names_the_failure_class_never_its_text(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    section: str,
    reads: list[tuple[type, str]],
) -> None:
    for fake, method in reads:
        monkeypatch.setattr(fake, method, _leak)
    page = _facts(_archived(tmp_path, EVERY_SERVICE), tmp_path)

    answer = await page.pv_facts("SR:BPM1")
    served = _sections(answer)[section]

    assert served["outcome"] == "unavailable"
    assert served["reason"] == "the service failed: _Leak"
    assert "service.example" not in json.dumps(answer) and "secret" not in json.dumps(answer)


async def test_a_raising_section_reader_names_the_failure_class_never_its_text(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    page = _facts(_config(tmp_path, LIVE_BUILD, EVERY_SERVICE), tmp_path)
    monkeypatch.setattr(page._service, "snapshot", _leak)

    served = _sections(await page.pv_facts("SR:BPM1"))["alarm_state"]

    assert served["reason"] == "the service failed: _Leak"
    assert "secret" not in json.dumps(served)


@pytest.mark.usefixtures("archiver")
async def test_an_archive_on_another_target_says_not_available(
    tmp_path: Path, record: _Record
) -> None:
    record.move("va", 1)
    config = _config(tmp_path, VA_BUILD, {})
    config["archiver"] = APPLIANCE
    page = _facts(config, tmp_path)

    section = _sections(await page.pv_facts("SR:BPM1"))["archive"]

    assert section["outcome"] == "unavailable"
    assert section["reason"] == _connectors.NO_SOURCE_REASON


async def test_the_archive_summary_counts_the_directory_per_ioc(
    tmp_path: Path, archiver: type[FakeArchiver]
) -> None:
    archiver.held = ArchiveExtent(
        addresses=frozenset({"SR:BPM1", "SR:RF1", "SR:GONE"}),
        oldest_at=datetime(2026, 10, 13, tzinfo=UTC),
        newest_at=datetime(2026, 11, 12, 8, 0, 4, tzinfo=UTC),
    )
    page = _facts(_archived(tmp_path), tmp_path)

    summary = await page.archive_summary()

    assert summary["outcome"] == "found"
    assert summary["archived"] == {"count": 2, "of": 3}
    assert summary["by_ioc"] == [
        {"ioc": "ioc-a", "pvs": 2, "archived": 1},
        {"ioc": "ioc-b", "pvs": 1, "archived": 1},
    ]
    assert "by_ioc_reason" not in summary
    assert summary["newest_sample"] == "2026-11-12T08:00:04+00:00"
    assert summary["oldest_sample"] == "2026-10-13T00:00:00+00:00"
    assert summary["freshness_bound_s"] is None
    assert summary["configuration"] == {"archiver": {"type": "epics_archiver"}}
    assert summary["targets"] == [{"target": "live", "archive": True}]


@pytest.mark.usefixtures("archiver")
async def test_an_archive_that_cannot_list_its_channels_counts_nothing(tmp_path: Path) -> None:
    page = _facts(_archived(tmp_path), tmp_path)

    summary = await page.archive_summary()

    assert summary["archived"] == {"count": None, "of": 3, "reason": "this archiver does not say"}
    assert summary["by_ioc"] == []
    assert summary["by_ioc_reason"] == "this archiver does not say"
    assert summary["newest_sample"] is None


async def test_above_the_filter_cap_the_summary_gives_the_caps_reason(
    tmp_path: Path, archiver: type[FakeArchiver]
) -> None:
    archiver.held = ArchiveExtent(addresses=frozenset({"SR:BPM1"}))
    page = _facts(_archived(tmp_path, filter_cap=2), tmp_path)

    summary = await page.archive_summary()

    reason = "too many results to filter (3); narrow the search"
    assert summary["archived"] == {"count": None, "of": 3, "reason": reason}
    assert summary["by_ioc_reason"] == reason


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
        assert section["caveats"] == [], key
    assert {field: ioc[field] for field in expected} == expected
    assert ioc["health_pvs"] == []
