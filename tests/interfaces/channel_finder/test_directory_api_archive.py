"""The ``archive`` route under ``/api/directory``: what the current target's archive holds.

Every test runs the real lifespan and the real ``DirectoryService`` over a config,
with the directory backend and the archiver connector swapped for fakes. The
control target comes from a stubbed control-context record.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("osprey_connectors.service_answers")

from fastapi.testclient import TestClient

import osprey.channel_roster as channel_roster
from osprey_connectors import control_context
from osprey_connectors.archiver import ArchiveExtent
from osprey_connectors.directory import DirectoryEntry, DirectoryPage
from osprey_connectors.service_answers import Source
from tests._service_factory import patch_create_service

_CONFIG_SEAM = "osprey_connectors.workspace.load_osprey_config"
_FIXTURE = Path(__file__).parent / "fixtures" / "directory_routes.json"

VA_BUILD = {"type": "virtual_accelerator"}
FACILITY = {"type": "channelfinder", "settings": {"url": "http://cf.example"}}
OWN_STORE = {
    "type": "mongodb_archiver",
    "mongodb_archiver": {
        "host": "localhost",
        "port": 27801,
        "name": "osprey_archiver",
        "collection": "pv_history",
        "auth": {"source": "admin", "username": "osprey", "password_env": "MONGO_PASSWORD"},
    },
}
KNOBS = {
    "retention_days": 30,
    "hot_span_hours": 48,
    "hot_cadence_sec": 10,
    "tail_cadence_sec": 60,
    "recorder_cadence_sec": 10,
}


class FakeDirectory:
    """A directory of three PVs on two IOCs."""

    rows = (
        DirectoryEntry(address="SR:BPM1", server="sim-SR"),
        DirectoryEntry(address="SR:BPM2", server="sim-SR"),
        DirectoryEntry(address="BR:BPM1", server="sim-BR"),
    )

    def __init__(self, spec: Any, **seams: Any) -> None:
        self.spec = spec

    @property
    def source(self) -> Any:
        return Source(kind="directory", target=self.spec.target, service="fake", rehearsal=False)

    field_properties: dict[str, str] = {}

    async def search(self, query: Any, *, timeout_s: float) -> Any:  # noqa: ARG002 - a fake
        rows = list(self.rows)[query.page * query.page_size :][: query.page_size]
        return DirectoryPage(
            outcome="found",
            searched="",
            request="GET /fake",
            source=self.source,
            total=len(self.rows),
            rows=rows,
            truncated=False,
        )

    async def count(self, query: Any, *, timeout_s: float) -> Any:  # noqa: ARG002 - a fake
        return len(self.rows)


class FakeStore:
    """The deployment's own store, recording two of the three PVs."""

    def __init__(self, spec: Any) -> None:
        self.source = Source(
            kind=spec.kind, target=spec.target, service=spec.type, rehearsal=spec.rehearsal
        )

    async def extent(self, **_kwargs: Any) -> Any:
        return ArchiveExtent(
            addresses=frozenset({"SR:BPM1", "BR:BPM1"}),
            oldest_at=datetime(2026, 10, 13, 8, tzinfo=UTC),
            newest_at=datetime(2026, 11, 12, 8, 0, 4, tzinfo=UTC),
        )

    async def aclose(self) -> None:
        pass


@pytest.fixture(autouse=True)
def fakes(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    async def directory(spec: Any, **seams: Any) -> FakeDirectory:
        return FakeDirectory(spec, **seams)

    async def archiver(spec: Any, **_kwargs: Any) -> FakeStore:
        return FakeStore(spec)

    patch_create_service(monkeypatch, "directory", directory)
    patch_create_service(monkeypatch, "archive", archiver)
    monkeypatch.setenv("OSPREY_CONFIG", str(tmp_path / "config.yml"))
    record = control_context.ControlContext(target="va", generation=1)
    monkeypatch.setattr(control_context, "read_record", lambda **_: record)
    channel_roster._roster_cache.clear()


def _config(tmp_path: Path, **extra: Any) -> dict[str, Any]:
    facility = {"identity": {"code": "test", "name": "Test"}, "channels": []}
    (tmp_path / "facility.json").write_text(json.dumps(facility), encoding="utf-8")
    config: dict[str, Any] = {
        "config_dir": str(tmp_path),
        "channel_finder": {
            "pipeline_mode": "in_context",
            "pipelines": {"in_context": {"database": {"path": "/tmp/db.json", "type": "flat"}}},
        },
        "control_system": VA_BUILD,
        "services": {"virtual_accelerator": {"port": 5064}},
        "pv_directory": {},
    }
    config.update(extra)
    return config


@pytest.fixture()
def start(mock_registry, monkeypatch: pytest.MonkeyPatch):  # noqa: ARG001
    @contextmanager
    def started(config: dict[str, Any]) -> Iterator[TestClient]:
        monkeypatch.setattr(_CONFIG_SEAM, lambda: config)
        from osprey.interfaces.channel_finder.app import create_app

        with TestClient(create_app(project_cwd="/tmp/test-project")) as client:
            yield client

    return started


def _shape(value: Any) -> Any:
    """*value*'s keys all the way down: what a fixture and an answer must share."""
    if isinstance(value, dict):
        return {key: _shape(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_shape(value[0])] if value else []
    return None


def test_the_route_answers_the_shared_fixtures_shape(start, tmp_path: Path) -> None:
    with start(_config(tmp_path, archiver=OWN_STORE, va_archiver=KNOBS)) as client:
        response = client.get("/api/directory/archive")

    assert response.status_code == 200
    answer = response.json()
    shared = json.loads(_FIXTURE.read_text(encoding="utf-8"))["archive"]
    assert _shape(answer) == _shape(shared)
    assert answer["archived"] == {"count": 2, "of": 3}
    assert answer["by_ioc"] == [
        {"ioc": "sim-SR", "pvs": 2, "archived": 1},
        {"ioc": "sim-BR", "pvs": 1, "archived": 1},
    ]
    assert answer["newest_sample"] == "2026-11-12T08:00:04+00:00"
    assert answer["configuration"] == {
        "archiver": {"type": "mongodb_archiver"},
        "va_archiver": KNOBS,
    }
    assert "MONGO_PASSWORD" not in response.text


def test_without_an_archive_on_the_target_it_says_so(start, tmp_path: Path) -> None:
    with start(_config(tmp_path)) as client:
        answer = client.get("/api/directory/archive").json()

    assert answer["outcome"] == "unavailable"
    assert answer["reason"] == "not available for this target"
    assert answer["targets"] == [{"target": "va", "archive": False}]


def test_the_route_is_read_only(start, tmp_path: Path) -> None:
    with start(_config(tmp_path, archiver=OWN_STORE)) as client:
        for method in ("post", "put", "patch", "delete"):
            assert getattr(client, method)("/api/directory/archive").status_code == 405
