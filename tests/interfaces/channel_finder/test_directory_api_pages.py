"""The PV page, IOC page and Logbook availability routes under ``/api/directory``.

Every test runs the real lifespan and the real ``DirectoryService`` over a
config, with the factory's create calls swapped for fakes. The control target
comes from a stubbed control-context record, or the deployment's baseline.
"""

from __future__ import annotations

import ast
import fnmatch
import importlib
import inspect
import json
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from urllib.parse import quote

import pytest

pytest.importorskip("osprey_connectors.service_answers")

from fastapi import FastAPI
from fastapi.testclient import TestClient

import osprey.channel_roster as channel_roster
from osprey.interfaces.channel_finder import directory_pages_api
from osprey.services.pv_directory import _connectors, service
from osprey_connectors import control_context
from osprey_connectors.alarm_state import AlarmConfigRecord
from osprey_connectors.directory import DirectoryPage, DirectoryRecord, DirectoryRow
from osprey_connectors.put_log import PutRow
from osprey_connectors.service_answers import HistoryPage, Outcome, Source
from tests._service_factory import patch_create_service

_CONFIG_SEAM = "osprey_connectors.workspace.load_osprey_config"
_FIXTURE = Path(__file__).parent / "fixtures" / "directory_routes.json"
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
_NO_SOURCE = "not available for this target"

VA_BUILD = {"type": "virtual_accelerator"}
LIVE_BUILD = {"type": "epics"}
FACILITY = {"type": "channelfinder", "settings": {"url": "http://cf.example"}}
ALARM_LOG = {"type": "phoebus_alarm_logger", "settings": {"url": "http://al.example"}}
PUT_LOG = {"type": "caputlog", "settings": {"url": "http://pl.example"}}
ARIEL = {"database": {"url": "postgresql://ariel.example/logbook"}}


# -- fakes --------------------------------------------------------------------


class FakeBackend:
    def __init__(self, spec: Any, **seams: Any) -> None:
        self.spec = spec

    @property
    def source(self) -> Any:
        return Source(
            kind=self.spec.kind,
            target=self.spec.target,
            service=self.spec.type,
            rehearsal=self.spec.rehearsal,
        )

    async def probe(self, deadline: float) -> Any:  # noqa: ARG002 - a fake
        return "answered"


class FakeDirectory(FakeBackend):
    """A directory over fixed records: ``ioc`` and ``status`` exact, ``name`` a glob."""

    records: dict[str, Any] = {}

    def _hits(self, query: Any) -> list[Any]:
        return [
            record.row
            for record in self.records.values()
            if (not query.server or record.row.server in query.server)
            and (query.status is None or record.row.status == query.status)
            and (query.name is None or fnmatch.fnmatchcase(record.row.name, query.name))
            and query.words in record.row.name
        ]

    async def search(self, query: Any, deadline: float) -> Any:  # noqa: ARG002 - a fake
        hits = self._hits(query)
        start = query.page * query.page_size
        rows = hits[start : start + query.page_size]
        return DirectoryPage(
            outcome=Outcome.FOUND if hits else Outcome.NO_MATCH,
            searched="",
            request="GET /fake",
            source=self.source,
            total=len(hits),
            rows=rows,
            truncated=start + len(rows) < len(hits),
        )

    async def count(self, query: Any, deadline: float) -> Any:  # noqa: ARG002 - a fake
        return len(self._hits(query))

    async def describe(self, name: str, deadline: float) -> Any:  # noqa: ARG002 - a fake
        return self.records.get(name, Outcome.NOT_LISTED)


class FakeHistory(FakeBackend):
    async def history(self, pv: str, since: Any, limit: int, deadline: float) -> Any:  # noqa: ARG002
        return HistoryPage(
            outcome=Outcome.FOUND,
            source=self.source,
            rows=[PutRow(at="t", channel=pv, old="1", new="2")],
        )

    async def config_for(self, pv: str, deadline: float) -> Any:  # noqa: ARG002 - a fake
        return AlarmConfigRecord(in_tree="yes", path=f"/SR/{pv}")


_FAKES = {"pv_directory": FakeDirectory, "alarm_log": FakeHistory, "put_log": FakeHistory}


@pytest.fixture(autouse=True)
def fakes(monkeypatch: pytest.MonkeyPatch) -> None:
    def create(kind: str):
        async def create_connector(spec: Any, *, project_dir: Path | None = None, **seams: Any):
            return _FAKES[kind](spec, project_dir=project_dir, **seams)

        return create_connector

    for kind in _FAKES:
        spec_kind = "directory" if kind == "pv_directory" else kind
        patch_create_service(monkeypatch, spec_kind, create(kind))
    monkeypatch.setattr(FakeDirectory, "records", {})


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
def _config_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("OSPREY_CONFIG", str(tmp_path / "config.yml"))


def _config(
    tmp_path: Path,
    control_system: dict[str, Any],
    section: dict[str, Any],
    *curated: str,
    **extra: Any,
) -> dict[str, Any]:
    channels = [{"id": address, "role": "none"} for address in curated]
    (tmp_path / "facility.json").write_text(
        json.dumps({"identity": {"code": "test", "name": "Test"}, "channels": channels}),
        encoding="utf-8",
    )
    config: dict[str, Any] = {
        "config_dir": str(tmp_path),
        "channel_finder": {
            "pipeline_mode": "in_context",
            "pipelines": {"in_context": {"database": {"path": "/tmp/db.json", "type": "flat"}}},
        },
        "control_system": control_system,
        "pv_directory": section,
    }
    if control_system == VA_BUILD:
        config["services"] = {"virtual_accelerator": {"port": 5064}}
    config.update(extra)
    return config


@pytest.fixture()
def start(mock_registry, monkeypatch: pytest.MonkeyPatch):  # noqa: ARG001
    """Create the app over a config and enter its lifespan."""

    @contextmanager
    def started(config: dict[str, Any]) -> Iterator[tuple[FastAPI, TestClient]]:
        monkeypatch.setattr(_CONFIG_SEAM, lambda: config)
        from osprey.interfaces.channel_finder.app import create_app

        app = create_app(project_cwd="/tmp/test-project")
        with TestClient(app) as client:
            yield app, client

    return started


def _record_of(name: str, ioc: str = "ioc-a", **properties: str) -> Any:
    row = DirectoryRow(name=name, server=ioc, host="host-a", status="Active")
    return DirectoryRecord(row=row, properties=properties)


def _records(*records: Any) -> dict[str, Any]:
    return {record.row.name: record for record in records}


def _routes() -> dict[str, Any]:
    return json.loads(_FIXTURE.read_text(encoding="utf-8"))


def _get(client: TestClient, route: str, name: str) -> dict[str, Any]:
    response = client.get(f"/api/directory/{route}/{quote(name, safe='')}")
    assert response.status_code == 200
    return response.json()


def _sections(answer: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {section["section"]: section for section in answer["sections"]}


# -- the PV page --------------------------------------------------------------


def test_the_pv_page_answers_every_configured_section(start, tmp_path: Path) -> None:
    FakeDirectory.records = _records(_record_of("SR:BPM1", recordDesc="beam position"))
    section = {**FACILITY, "alarm_log": ALARM_LOG, "put_log": PUT_LOG}
    with start(_config(tmp_path, LIVE_BUILD, section, "SR:BPM1")) as (_app, client):
        answer = _get(client, "pv", "SR:BPM1")

    assert answer["name"] == "SR:BPM1"
    sections = _sections(answer)
    assert list(sections) == ["directory", "alarm_config", "alarm_history", "put_history"]
    assert sections["alarm_config"]["record"]["in_tree"] == "yes"
    assert sections["directory"]["outcome"] == "found"
    assert sections["directory"]["origin"] == "curated"
    assert sections["put_history"]["outcome"] == "found"


def test_an_unconfigured_section_is_absent_from_the_pv_page(start, tmp_path: Path) -> None:
    FakeDirectory.records = _records(_record_of("SR:BPM1"))
    with start(_config(tmp_path, LIVE_BUILD, FACILITY)) as (_app, client):
        sections = _sections(_get(client, "pv", "SR:BPM1"))

    assert list(sections) == ["directory"]


def test_a_section_with_no_source_on_the_target_says_so(
    start, tmp_path: Path, record: _Record
) -> None:
    record.move("va", 1)
    section = {**FACILITY, "put_log": PUT_LOG}
    with start(_config(tmp_path, VA_BUILD, section)) as (_app, client):
        sections = _sections(_get(client, "pv", "SR:BPM1"))

    assert sections["put_history"]["outcome"] == "unavailable"
    assert sections["put_history"]["reason"] == _NO_SOURCE
    assert sections["put_history"]["source"]["target"] == "va"


def test_the_pv_page_matches_the_shared_fixture(start, tmp_path: Path) -> None:
    FakeDirectory.records = _records(_record_of("SR:BPM1", recordDesc="beam position"))
    section = {**FACILITY, "alarm_log": ALARM_LOG, "put_log": PUT_LOG}
    with start(_config(tmp_path, LIVE_BUILD, section, "SR:BPM1")) as (_app, client):
        answer = _get(client, "pv", "SR:BPM1")

    shared = _routes()["pv/{name}"]
    assert set(answer) == set(shared)
    expected = _sections(shared)
    # The fixture's alarm history is unavailable; a found history has put_history's shape.
    expected["alarm_history"] = {**expected["put_history"], "section": "alarm_history"}
    for key, got in _sections(answer).items():
        assert set(got) == set(expected[key]), key
    assert set(_sections(answer)["directory"]["record"]) == set(expected["directory"]["record"])


def test_the_name_is_one_percent_encoded_segment(start, tmp_path: Path) -> None:
    name = "SR:BPM1/X Y"
    FakeDirectory.records = _records(_record_of(name))
    with start(_config(tmp_path, LIVE_BUILD, FACILITY)) as (_app, client):
        answer = _get(client, "pv", name)

    assert answer["name"] == name
    assert _sections(answer)["directory"]["outcome"] == "found"


# -- the IOC page -------------------------------------------------------------


def _live_ioc_records() -> dict[str, Any]:
    return _records(
        _record_of("SR:BPM1", caPort="5064", pvaPort="5075", Engineer="A. Engineer"),
        _record_of("SR:IOC-A:HEARTBEAT"),
        _record_of("SR:OTHER", ioc="ioc-b"),
    )


def test_the_ioc_page_payload(start, tmp_path: Path) -> None:
    FakeDirectory.records = _live_ioc_records()
    with start(_config(tmp_path, LIVE_BUILD, FACILITY, "SR:BPM1")) as (_app, client):
        ioc = _get(client, "ioc", "ioc-a")

    assert ioc["ioc"] == "ioc-a"
    assert ioc["outcome"] == "found"
    assert ioc["host"] == "host-a"
    assert ioc["ports"] == [5064, 5075]
    assert ioc["pv_count"] == 2
    assert ioc["curated_count"] == 1
    assert ioc["engineer"] == "A. Engineer"
    assert ioc["engineer_source"] == "directory property"
    assert ioc["health_pvs"] == ["SR:IOC-A:HEARTBEAT"]
    assert ioc["links"] == {"panel": "/panel/channel-finder/#/ioc/ioc-a"}


def test_the_ioc_page_matches_the_shared_fixture(start, tmp_path: Path) -> None:
    FakeDirectory.records = _live_ioc_records()
    with start(_config(tmp_path, LIVE_BUILD, FACILITY, "SR:BPM1")) as (_app, client):
        ioc = _get(client, "ioc", "ioc-a")

    assert set(ioc) == set(_routes()["ioc/{name}"])


def test_an_ioc_name_is_one_percent_encoded_segment(start, tmp_path: Path) -> None:
    FakeDirectory.records = _records(_record_of("SR:BPM1", ioc="ioc-a/1"))
    with start(_config(tmp_path, LIVE_BUILD, FACILITY)) as (_app, client):
        ioc = _get(client, "ioc", "ioc-a/1")

    assert ioc["ioc"] == "ioc-a/1"
    assert ioc["pv_count"] == 1
    assert ioc["links"]["panel"].endswith("#/ioc/ioc-a%2F1")


# -- Logbook availability -----------------------------------------------------


def _with_ariel(config: dict[str, Any]) -> dict[str, Any]:
    return {**config, "ariel": ARIEL, "web": {"panels": {"ariel": True}}}


def test_the_logbook_is_not_available_on_the_va(start, tmp_path: Path, record: _Record) -> None:
    record.move("va", 1)
    config = _with_ariel(_config(tmp_path, VA_BUILD, FACILITY))
    with start(config) as (_app, client):
        assert client.get("/api/info").json()["directory"]["ariel"] is True
        answer = client.get("/api/directory/logbook/availability").json()

    assert answer == {"available": False, "source": None, "reason": _NO_SOURCE}


def test_the_logbook_on_live_follows_the_descriptor(start, tmp_path: Path) -> None:
    with start(_with_ariel(_config(tmp_path, LIVE_BUILD, FACILITY))) as (_app, client):
        ariel = client.get("/api/info").json()["directory"]["ariel"]
        answer = client.get("/api/directory/logbook/availability").json()

    assert ariel is True
    assert answer == _routes()["logbook/availability"]

    with start(_config(tmp_path, LIVE_BUILD, FACILITY)) as (_app, client):
        ariel = client.get("/api/info").json()["directory"]["ariel"]
        answer = client.get("/api/directory/logbook/availability").json()

    assert ariel is False
    assert answer == {"available": False, "source": None, "reason": _NO_SOURCE}


def test_the_logbook_follows_a_target_move(start, tmp_path: Path, record: _Record) -> None:
    config = _with_ariel(_config(tmp_path, LIVE_BUILD, FACILITY))
    config["services"] = {"virtual_accelerator": {"port": 5064}}
    with start(config) as (_app, client):
        assert client.get("/api/directory/logbook/availability").json()["available"] is True
        record.move("va", 1)
        assert client.get("/api/directory/logbook/availability").json()["available"] is False


@pytest.fixture()
def floor() -> Iterator[None]:
    """The shim, the service and this route module re-imported without the new modules."""
    with pytest.MonkeyPatch.context() as patch:
        for name in _NEW_MODULES:
            patch.setitem(sys.modules, name, None)
        importlib.reload(_connectors)
        importlib.reload(service)
        importlib.reload(directory_pages_api)
        yield
    importlib.reload(_connectors)
    importlib.reload(service)
    importlib.reload(directory_pages_api)


@pytest.mark.usefixtures("floor")
def test_on_the_floor_the_logbook_answers_the_upgrade_reason(start, tmp_path: Path) -> None:
    with start(_with_ariel(_config(tmp_path, LIVE_BUILD, FACILITY))) as (_app, client):
        answer = client.get("/api/directory/logbook/availability").json()

    assert answer == {"available": False, "source": None, "reason": _connectors.FLOOR_REASON}


def test_no_ariel_client_is_reachable_from_the_module() -> None:
    tree = ast.parse(inspect.getsource(directory_pages_api))
    imported = [
        alias.name if isinstance(node, ast.Import) else f"{node.module}.{alias.name}"
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    ]

    assert imported
    assert not [name for name in imported if "ariel" in name.lower() or "httpx" in name]
