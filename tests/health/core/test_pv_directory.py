"""Tests for the ``pv_directory`` and ``pv_directory_alarm_feed`` health categories.

Both read the directory service's probes and build no backend: the backends here
come from the factory's create API, swapped for fakes, a Kafka feed over a fake
consumer, or the packaged replay. The control target is the deployment's baseline.
"""

from __future__ import annotations

import asyncio
import importlib
import json
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from osprey.health.core.pv_directory import pv_directory, pv_directory_alarm_feed
from osprey.health.models import Status

LIVE_BUILD = {"type": "epics"}
VA_BUILD = {"type": "virtual_accelerator"}
FACILITY = {"type": "channelfinder", "settings": {"url": "http://cf.example"}}
KAFKA = {
    "type": "phoebus_alarm_server",
    "settings": {"bootstrap_servers": "k:9092", "topic": "Accelerator", "timeout_s": 0.2},
}
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


@pytest.fixture
def contract() -> Any:
    return pytest.importorskip("osprey_connectors.service_answers")


@pytest.fixture(autouse=True)
def no_record(monkeypatch: pytest.MonkeyPatch) -> None:
    from osprey_connectors import control_context

    monkeypatch.setattr(control_context, "read_record", lambda **_: None)


def _config(tmp_path: Path, control_system: dict[str, Any], section: Any) -> dict[str, Any]:
    (tmp_path / "facility.json").write_text(json.dumps({"channels": []}), encoding="utf-8")
    config: dict[str, Any] = {"config_dir": str(tmp_path), "control_system": control_system}
    if control_system == VA_BUILD:
        config["services"] = {"virtual_accelerator": {"port": 5064}}
    if section is not None:
        config["pv_directory"] = section
    return config


class _Fake:
    """A backend whose probe answers *answer* at once; ``None`` answers its type."""

    answers: dict[str, Any] = {}

    def __init__(self, spec: Any, *, project_dir: Path | None = None, **_: Any) -> None:
        self.spec = spec
        self.project_dir = project_dir

    @property
    def source(self) -> Any:
        contract = importlib.import_module("osprey_connectors.service_answers")
        return contract.Source(
            kind=self.spec.kind, target=self.spec.target, service=self.spec.type, rehearsal=False
        )

    async def probe(self, deadline: float) -> Any:  # noqa: ARG002 - a fake
        return self.answers.get(self.spec.kind, f"{self.spec.type} answered")


@pytest.fixture
def fakes(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    """Fakes behind the directory, alarm logger and put log create calls."""
    pytest.importorskip("osprey_connectors.service_answers")
    from osprey_connectors.factory import ConnectorFactory

    made: list[Any] = []

    async def create(spec: Any, *, project_dir: Path | None = None, **_: Any) -> Any:
        made.append(_Fake(spec, project_dir=project_dir))
        return made[-1]

    for kind in ("pv_directory", "alarm_log", "put_log"):
        monkeypatch.setattr(ConnectorFactory, f"create_{kind}_connector", create)
    monkeypatch.setattr(_Fake, "answers", {})
    return made


class _Consumer:
    """An aiokafka-shaped consumer over one partition whose end offset is *end*.

    It delivers nothing, so with ``end > 0`` the feed never catches up.
    """

    def __init__(self, end: int) -> None:
        self.end = end

    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        pass

    def assignment(self) -> set[str]:
        return {"p0"}

    async def end_offsets(self, partitions: Any) -> dict[str, int]:
        return dict.fromkeys(partitions, self.end)

    async def seek_to_beginning(self, *_: Any) -> None:
        pass

    async def position(self, _partition: Any) -> int:
        return 0

    async def getmany(self, *_: Any, **__: Any) -> dict[str, list[Any]]:
        await asyncio.sleep(0.01)
        return {}


def _kafka(monkeypatch: pytest.MonkeyPatch, consumer: Any | None) -> None:
    """The alarm-state create call builds the Kafka backend over *consumer*."""
    kafka = pytest.importorskip("osprey_connectors.alarm_state._phoebus_kafka")
    from osprey_connectors.factory import ConnectorFactory

    async def create(spec: Any, *, project_dir: Path | None = None, **_: Any) -> Any:
        factory = None if consumer is None else (lambda _settings: consumer)
        return kafka.PhoebusAlarmState(spec, project_dir=project_dir, consumer_factory=factory)

    monkeypatch.setattr(ConnectorFactory, "create_alarm_state_connector", create)


def _alarm_tasks() -> list[asyncio.Task[Any]]:
    current = asyncio.current_task()
    return [
        task
        for task in asyncio.all_tasks()
        if task is not current and task.get_name().startswith(("alarm-", "phoebus-"))
    ]


# -- off mode -----------------------------------------------------------------


@pytest.mark.parametrize("factory", [pv_directory, pv_directory_alarm_feed])
async def test_no_config_gives_no_rows(factory: Any) -> None:
    assert await factory(None)() == []


@pytest.mark.parametrize("factory", [pv_directory, pv_directory_alarm_feed])
async def test_an_unconfigured_directory_gives_no_rows(factory: Any, tmp_path: Path) -> None:
    assert await factory(_config(tmp_path, LIVE_BUILD, None), cwd=tmp_path)() == []


# -- the poll category ----------------------------------------------------------


async def test_each_configured_service_is_one_row_with_its_evidence(
    tmp_path: Path, fakes: list[Any]
) -> None:
    _Fake.answers = {"alarm_log": "root answered (alarms); this does not prove ingestion"}
    section = {
        **FACILITY,
        "alarm_log": {"type": "phoebus_alarm_logger", "settings": {"url": "http://al"}},
        "put_log": {"type": "caputlog", "settings": {"url": "http://pl"}},
        "alarm_state": KAFKA,
    }

    rows = await pv_directory(_config(tmp_path, LIVE_BUILD, section), cwd=tmp_path)()

    assert [(r.name, r.status) for r in rows] == [
        ("pv_directory.directory", Status.OK),
        ("pv_directory.alarm_log", Status.OK),
        ("pv_directory.put_log", Status.OK),
    ]
    assert rows[0].value == "channelfinder answered"
    assert rows[1].value == "root answered (alarms); this does not prove ingestion"
    assert all(r.category == "pv_directory" and r.probed for r in rows)
    assert rows[0].details == "channelfinder on live"
    assert all(backend.project_dir == tmp_path for backend in fakes)
    assert _alarm_tasks() == []


@pytest.mark.usefixtures("fakes")
async def test_an_unreachable_service_warns_with_its_reason(tmp_path: Path, contract: Any) -> None:
    _Fake.answers = {"directory": contract.Unavailable("the service answered a login page")}

    rows = await pv_directory(_config(tmp_path, LIVE_BUILD, FACILITY), cwd=tmp_path)()

    assert len(rows) == 1
    assert rows[0].status == Status.WARNING
    assert rows[0].message == "directory: the service answered a login page"


@pytest.mark.usefixtures("fakes")
async def test_a_service_with_no_source_on_the_target_is_skipped(tmp_path: Path) -> None:
    section = {
        "alarm_log": {"type": "phoebus_alarm_logger", "settings": {"url": "http://al"}},
    }

    rows = await pv_directory(_config(tmp_path, VA_BUILD, section), cwd=tmp_path)()

    alarm_log = next(r for r in rows if r.name == "pv_directory.alarm_log")
    assert alarm_log.status == Status.SKIP
    assert alarm_log.message == "alarm logger: not available for this target"


# -- the alarm feed -------------------------------------------------------------


@pytest.mark.usefixtures("contract")
async def test_a_silent_feed_answers_loading_once_the_deadline_ran_out(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _kafka(monkeypatch, _Consumer(end=3))
    config = _config(tmp_path, LIVE_BUILD, {**FACILITY, "alarm_state": KAFKA})

    rows = await pv_directory_alarm_feed(config, cwd=tmp_path)()

    assert [(r.name, r.category, r.status) for r in rows] == [
        ("pv_directory.alarm_feed", "pv_directory_alarm_feed", Status.WARNING)
    ]
    assert rows[0].message == "alarm feed: alarm feed still loading"
    assert _alarm_tasks() == []


@pytest.mark.usefixtures("contract")
async def test_a_feed_that_caught_up_shows_the_backend_s_words(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _kafka(monkeypatch, _Consumer(end=0))
    config = _config(tmp_path, LIVE_BUILD, {**FACILITY, "alarm_state": KAFKA})

    rows = await pv_directory_alarm_feed(config, cwd=tmp_path)()

    assert rows[0].status == Status.OK
    assert rows[0].value == "read 0 messages; last contact 0 s ago"
    assert rows[0].details == "phoebus_alarm_server on live"
    assert _alarm_tasks() == []


@pytest.mark.usefixtures("contract")
async def test_a_missing_kafka_client_is_named(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    kafka = pytest.importorskip("osprey_connectors.alarm_state._phoebus_kafka")
    _kafka(monkeypatch, None)
    monkeypatch.setitem(sys.modules, "aiokafka", None)
    config = _config(tmp_path, LIVE_BUILD, {**FACILITY, "alarm_state": KAFKA})

    rows = await pv_directory_alarm_feed(config, cwd=tmp_path)()

    assert rows[0].status == Status.WARNING
    assert rows[0].message == f"alarm feed: {kafka.KAFKA_CLIENT_MISSING}"


@pytest.mark.usefixtures("contract")
async def test_the_replay_on_va_answers_at_once(tmp_path: Path) -> None:
    section = {"alarm_state": {"replay_file": "demo:example-alarms"}}

    rows = await pv_directory_alarm_feed(_config(tmp_path, VA_BUILD, section), cwd=tmp_path)()

    assert rows[0].status == Status.OK
    assert rows[0].value.endswith("messages from the replay file")
    assert rows[0].details == "alarm_replay on va"
    assert _alarm_tasks() == []


# -- the connectors floor -------------------------------------------------------


@pytest.fixture
def floor() -> Iterator[None]:
    """The directory service re-imported over a shim without the PV directory modules."""
    from osprey.services.pv_directory import _connectors, service

    with pytest.MonkeyPatch.context() as patch:
        for name in _NEW_MODULES:
            patch.setitem(sys.modules, name, None)
        importlib.reload(_connectors)
        importlib.reload(service)
        yield
    importlib.reload(_connectors)
    importlib.reload(service)


@pytest.mark.usefixtures("floor")
async def test_modules_blocked_one_row_names_the_upgrade(tmp_path: Path) -> None:
    from osprey.services.pv_directory._connectors import FLOOR_REASON

    config = _config(tmp_path, LIVE_BUILD, {**FACILITY, "alarm_state": KAFKA})

    rows = await pv_directory(config, cwd=tmp_path)()
    feed = await pv_directory_alarm_feed(config, cwd=tmp_path)()

    assert [(r.name, r.status) for r in rows] == [("pv_directory.directory", Status.WARNING)]
    assert rows[0].message == f"directory: {FLOOR_REASON}"
    assert feed == []
