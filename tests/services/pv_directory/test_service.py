"""Tests for the directory service, with fake backends behind the factory's create API.

The service is the only place a backend is built, so every test swaps the four
create calls for one that records the spec it was handed and returns a fake. The
control target comes from a stubbed control-context record, or from the
deployment's baseline when there is none.
"""

from __future__ import annotations

import asyncio
import dataclasses
import importlib
import json
import sys
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

pytest.importorskip("osprey_connectors.service_answers")

import osprey.channel_roster as channel_roster
from osprey.services.pv_directory import _connectors, service
from osprey.services.pv_directory.demo import resolve_replay_file
from osprey_connectors import control_context
from osprey_connectors.alarm_state import AlarmStateRecord, AlarmStateRow, AlarmStates
from osprey_connectors.directory import DirectoryPage, DirectoryRecord, DirectoryRow
from osprey_connectors.service_answers import Outcome, Source, Unavailable
from tests._service_factory import patch_create_service

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
FACILITY = {"type": "channelfinder", "settings": {"url": "http://cf.example", "timeout_s": 0.3}}
_ROUTES = Path(__file__).parents[2] / "interfaces/channel_finder/fixtures/directory_routes.json"


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

    @property
    def field_properties(self) -> Any:
        """The property map a ChannelFinder directory answers, built with the spec's settings."""
        description = self.spec.settings.get("description_property") or "recordDesc"
        return {
            "status": "pvStatus",
            "channel_type": "recordType",
            "server": "iocName",
            "host": "hostName",
            "last_announced": "time",
            "description": description,
        }

    async def probe(self, deadline: float) -> Any:  # noqa: ARG002 - a fake
        return f"{self.spec.type} answered"


def _row(name: str, **fields: Any) -> Any:
    return DirectoryRow(name=name, **fields)


class FakeDirectory(FakeBackend):
    """A directory over fixed rows; the name filter is a substring, ``ioc`` is exact."""

    rows: list[Any] = []
    records: dict[str, Any] = {}
    count_delay = 0.0
    search_delay = 0.0

    def __init__(self, spec: Any, **kwargs: Any) -> None:
        super().__init__(spec, **kwargs)
        self.searches: list[Any] = []
        self.counts: list[Any] = []

    def _hits(self, query: Any) -> list[Any]:
        return [
            row
            for row in self.rows
            if query.words in row.name and (not query.server or row.server in query.server)
        ]

    async def search(self, query: Any, deadline: float) -> Any:  # noqa: ARG002 - a fake
        self.searches.append(query)
        await asyncio.sleep(self.search_delay)
        hits = self._hits(query)
        start = query.page * query.page_size
        rows = hits[start : start + query.page_size]
        return DirectoryPage(
            outcome=Outcome.FOUND if hits else Outcome.NO_MATCH,
            searched="from the backend",
            request=f"GET /fake?q={query.words}&size={query.page_size}",
            source=self.source,
            total=len(hits),
            rows=rows,
            truncated=start + len(rows) < len(hits),
        )

    async def count(self, query: Any, deadline: float) -> Any:  # noqa: ARG002 - a fake
        self.counts.append(query)
        await asyncio.sleep(self.count_delay)
        return len(self._hits(query))

    async def describe(self, name: str, deadline: float) -> Any:  # noqa: ARG002 - a fake
        return self.records.get(name, Outcome.NOT_LISTED)


class FakeAlarmState(FakeBackend):
    """A feed holding ``held``: a state per channel, or ``Unavailable`` for the whole list."""

    held: Any = {}

    async def start(self) -> None:
        pass

    async def aclose(self) -> None:
        pass

    async def state_for(self, pv: str, deadline: float) -> Any:  # noqa: ARG002 - a fake
        return AlarmStateRecord(severity="MAJOR")

    async def config_for(self, pv: str, deadline: float) -> Any:  # noqa: ARG002 - a fake
        return Unavailable("no message from the feed yet")

    async def states(self, deadline: float) -> Any:  # noqa: ARG002 - a fake
        if isinstance(self.held, Unavailable):
            return self.held
        rows = tuple(
            AlarmStateRow(channel=channel, path=f"/Demo/{channel}", state=state)
            for channel, state in self.held.items()
        )
        return AlarmStates(rows=rows, stale=False, last_contact=None)


class FakeLog(FakeBackend):
    pass


_FAKES = {
    "pv_directory": FakeDirectory,
    "alarm_state": FakeAlarmState,
    "alarm_log": FakeLog,
    "put_log": FakeLog,
}


@pytest.fixture(autouse=True)
def built(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    """Every backend the service builds, in order; the create calls return fakes."""
    made: list[Any] = []

    def create(kind: str):
        async def create_connector(spec: Any, *, project_dir: Path | None = None, **seams: Any):
            assert spec.kind == ("directory" if kind == "pv_directory" else kind)
            backend = _FAKES[kind](spec, project_dir=project_dir, **seams)
            made.append(backend)
            return backend

        return create_connector

    for kind in _FAKES:
        spec_kind = "directory" if kind == "pv_directory" else kind
        patch_create_service(monkeypatch, spec_kind, create(kind))
    monkeypatch.setattr(FakeDirectory, "rows", [])
    monkeypatch.setattr(FakeDirectory, "records", {})
    monkeypatch.setattr(FakeDirectory, "count_delay", 0.0)
    monkeypatch.setattr(FakeDirectory, "search_delay", 0.0)
    monkeypatch.setattr(FakeAlarmState, "held", {})
    return made


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
) -> dict[str, Any]:
    channels = [{"id": address, "role": "none"} for address in curated]
    (tmp_path / "facility.json").write_text(json.dumps({"channels": channels}), encoding="utf-8")
    config: dict[str, Any] = {"config_dir": str(tmp_path), "control_system": control_system}
    if control_system == VA_BUILD:
        config["services"] = {"virtual_accelerator": {"port": 5064}}
    if section is not None:
        config["pv_directory"] = section
    return config


def _service(config: dict[str, Any], tmp_path: Path) -> Any:
    return service.DirectoryService(config, project_dir=tmp_path)


# -- search -------------------------------------------------------------------


async def test_search_marks_rows_and_removes_hidden_properties(tmp_path: Path) -> None:
    FakeDirectory.rows = [
        _row("SR:BPM1", server="ioc-a", host="h1", channel_type="ai"),
        _row("SR:BPM2", server="ioc-b", host="h2", channel_type="ao"),
    ]
    section = {**FACILITY, "display": {"hidden_properties": ["hostName"]}}
    directory = _service(_config(tmp_path, LIVE_BUILD, section, "SR:BPM1"), tmp_path)

    page = await directory.search(words="SR")

    assert page["outcome"] == "found"
    assert [(row["name"], row["origin"]) for row in page["rows"]] == [
        ("SR:BPM1", "curated"),
        ("SR:BPM2", "directory"),
    ]
    assert all(row["host"] is None for row in page["rows"])
    assert page["rows"][0]["ioc"] == "ioc-a"
    assert page["total"] == 2
    assert page["source"] == {
        "target": "live",
        "service": "channelfinder",
        "label": "All PVs",
        "rehearsal": False,
    }
    assert page["searched"] == "name or description matches SR"
    assert page["request"] == "GET /fake?q=SR&size=50"
    assert page["partial"] == []
    assert page["facets"]["origin"] == [
        {"value": "curated", "count": 1},
        {"value": "directory", "count": 1},
    ]
    assert page["facets"]["ioc"] == [
        {"value": "ioc-a", "count": 1},
        {"value": "ioc-b", "count": 1},
    ]
    assert page["facets_reason"] is None
    await directory.aclose()


async def test_search_filters_by_origin_over_the_fetched_set(
    tmp_path: Path, built: list[Any]
) -> None:
    FakeDirectory.rows = [_row(f"SR:PV{i}", server="ioc-a") for i in range(5)]
    config = _config(tmp_path, LIVE_BUILD, FACILITY, "SR:PV1", "SR:PV3")
    directory = _service(config, tmp_path)

    page = await directory.search(words="SR", origin=["curated"], page_size=1)

    assert [row["name"] for row in page["rows"]] == ["SR:PV1"]
    assert page["total"] == 2
    assert page["truncated"] is True
    assert page["searched"] == "name or description matches SR · origin curated"
    assert page["facets"]["origin"] == [
        {"value": "directory", "count": 3},
        {"value": "curated", "count": 2},
    ]
    sent = built[0].searches + built[0].counts
    assert sent
    assert all(not hasattr(query, "origin") for query in sent)
    await directory.aclose()


async def test_partial_on_a_slow_count(tmp_path: Path) -> None:
    FakeDirectory.rows = [_row("SR:BPM1")]
    FakeDirectory.count_delay = 5.0
    directory = _service(_config(tmp_path, LIVE_BUILD, FACILITY), tmp_path)

    page = await asyncio.wait_for(directory.search(words="SR"), 2.0)

    assert page["outcome"] == "found"
    assert [row["name"] for row in page["rows"]] == ["SR:BPM1"]
    assert {entry["part"] for entry in page["partial"]} == {"count", "facets"}
    assert page["facets"] is None
    assert page["facets_reason"]
    await directory.aclose()


async def test_above_the_filter_cap_facets_say_why(tmp_path: Path, built: list[Any]) -> None:
    FakeDirectory.rows = [_row(f"SR:PV{i}") for i in range(4)]
    section = {**FACILITY, "display": {"filter_cap": 3}}
    directory = _service(_config(tmp_path, LIVE_BUILD, section), tmp_path)

    page = await directory.search(words="SR", page_size=2)
    filtered = await directory.search(words="SR", origin=["curated"])

    assert page["facets"] is None
    assert page["facets_reason"] == "too many results to filter (4); narrow the search"
    assert [row["name"] for row in page["rows"]] == ["SR:PV0", "SR:PV1"]
    assert all(query.page_size <= 3 for query in built[0].searches)
    assert filtered["outcome"] == "unavailable"
    assert filtered["reason"] == "too many results to filter (4); narrow the search"
    await directory.aclose()


async def test_origin_search_without_its_set_has_no_total(tmp_path: Path) -> None:
    FakeDirectory.rows = [_row(f"SR:PV{i}") for i in range(4)]
    section = {**FACILITY, "display": {"filter_cap": 3}}
    directory = _service(_config(tmp_path, LIVE_BUILD, section), tmp_path)

    filtered = await directory.search(words="SR", origin=["curated"])

    assert filtered["outcome"] == "unavailable"
    assert filtered["total"] is None
    await directory.aclose()


@pytest.mark.parametrize(
    ("display", "cap"),
    [
        ({"filter_cap": 3}, 3),
        (None, 5000),
        ({"filter_cap": -1}, 5000),
        ({"filter_cap": True}, 5000),
    ],
)
def test_the_filter_cap_answers_display_filter_cap(
    tmp_path: Path, display: dict[str, Any] | None, cap: int
) -> None:
    section = dict(FACILITY) if display is None else {**FACILITY, "display": display}

    assert _service(_config(tmp_path, LIVE_BUILD, section), tmp_path).filter_cap == cap


async def test_a_zero_count_drops_the_backend_count_partial(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def search(self: Any, query: Any, deadline: float) -> Any:  # noqa: ARG001 - a fake
        return DirectoryPage(
            outcome=Outcome.NO_MATCH,
            searched="from the backend",
            request="GET /fake",
            source=self.source,
            total=None,
            rows=[],
            truncated=False,
            partial=[{"part": "count", "reason": "the count timed out"}],
        )

    monkeypatch.setattr(FakeDirectory, "search", search)
    directory = _service(_config(tmp_path, LIVE_BUILD, FACILITY), tmp_path)

    page = await directory.search(words="nothing")

    assert page["total"] == 0
    assert page["partial"] == []
    await directory.aclose()


class _Leak(ConnectionError):
    """A backend failure whose message carries a URL with a login."""


async def _leak(*_args: Any, **_kwargs: Any) -> Any:
    raise _Leak("cannot reach https://reader:secret@cf.example/api")


@pytest.mark.parametrize("method", ["search", "count"])
async def test_a_raising_search_names_the_failure_class_never_its_text(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, method: str
) -> None:
    monkeypatch.setattr(FakeDirectory, method, _leak)
    directory = _service(_config(tmp_path, LIVE_BUILD, FACILITY), tmp_path)

    page = await directory.search(words="BPM")

    reasons = [page.get("reason"), *(part["reason"] for part in page["partial"])]
    assert "the service failed: _Leak" in reasons
    assert "cf.example" not in json.dumps(page) and "secret" not in json.dumps(page)
    await directory.aclose()


async def test_a_raising_describe_names_the_failure_class_never_its_text(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(FakeDirectory, "describe", _leak, raising=False)
    directory = _service(_config(tmp_path, LIVE_BUILD, FACILITY), tmp_path)

    answer = await directory.describe("SR:BPM1")

    assert answer["reason"] == "the service failed: _Leak"
    assert "secret" not in json.dumps(answer)
    await directory.aclose()


async def test_search_past_its_deadline_leaves_no_task(tmp_path: Path) -> None:
    FakeDirectory.rows = [_row("SR:BPM1")]
    FakeDirectory.count_delay = 5.0
    directory = _service(_config(tmp_path, LIVE_BUILD, FACILITY), tmp_path)

    await asyncio.wait_for(directory.search(words="SR"), 2.0)

    others = {t for t in asyncio.all_tasks() if t is not asyncio.current_task()}
    assert not [t for t in others if not t.done()]
    await directory.aclose()


async def test_a_count_in_flight_across_a_target_move_is_not_cached(
    tmp_path: Path, built: list[Any], record: _Record
) -> None:
    FakeDirectory.rows = [_row("SR:BPM1")]
    FakeDirectory.count_delay = 0.2
    record.move("live", 1)
    directory = _service(_config(tmp_path, LIVE_BUILD, FACILITY), tmp_path)

    old = asyncio.create_task(directory.count(words="SR"))
    await asyncio.sleep(0.05)
    record.move("live", 2)
    await directory.backend("pv_directory")
    await old

    FakeDirectory.count_delay = 0.0
    await directory.count(words="SR")

    assert len(built) == 2
    assert len(built[1].counts) == 1
    await directory.aclose()


async def test_a_snapshot_built_across_aclose_is_not_installed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def slow_create(spec: Any, *, project_dir: Path | None = None, **seams: Any) -> Any:
        await asyncio.sleep(0.2)
        return FakeAlarmState(spec, project_dir=project_dir, **seams)

    patch_create_service(monkeypatch, "alarm_state", slow_create)
    section = {
        **FACILITY,
        "alarm_state": {
            "type": "phoebus_alarm_server",
            "settings": {"bootstrap_servers": "k:9092", "topic": "Accelerator"},
        },
    }
    directory = _service(_config(tmp_path, LIVE_BUILD, section), tmp_path)

    pending = asyncio.create_task(directory.snapshot())
    await asyncio.sleep(0.05)
    await directory.aclose()

    assert await pending is None
    assert directory._snapshot is None


async def test_count_answers_the_whole_directory(tmp_path: Path) -> None:
    FakeDirectory.rows = [_row("A"), _row("B")]
    directory = _service(_config(tmp_path, LIVE_BUILD, FACILITY, "A"), tmp_path)

    assert (await directory.count())["total"] == 2
    assert (await directory.count(origin=["curated"]))["total"] == 1
    assert (await directory.count(words="nothing"))["outcome"] == "no_match"
    await directory.aclose()


# -- the alarm join -------------------------------------------------------------

_FEED = {
    "type": "phoebus_alarm_server",
    "settings": {"bootstrap_servers": "k:9092", "topic": "Accelerator"},
}
_MAJOR = AlarmStateRecord(
    severity="MAJOR", status="HIHI_ALARM", latched=True, time="2026-03-02T08:12:30Z"
)


def _alarmed_directory(tmp_path: Path, **display: Any) -> Any:
    """Three rows: PV1 in MAJOR, PV2 held OK, PV3 with no state in the feed."""
    FakeDirectory.rows = [_row(f"SR:PV{i}", server="ioc-a") for i in (1, 2, 3)]
    FakeAlarmState.held = {"SR:PV1": _MAJOR, "SR:PV2": AlarmStateRecord(severity="OK")}
    section = {**FACILITY, "alarm_state": _FEED, "display": display}
    return _service(_config(tmp_path, LIVE_BUILD, section, "SR:PV1"), tmp_path)


async def test_search_marks_each_row_with_the_feed_state(tmp_path: Path) -> None:
    directory = _alarmed_directory(tmp_path)

    page = await directory.search(words="SR")

    assert [row["alarm"] for row in page["rows"]] == [
        {
            "severity": "MAJOR",
            "status": "HIHI_ALARM",
            "latched": True,
            "acknowledged": False,
            "time": "2026-03-02T08:12:30Z",
        },
        {"severity": "OK", "status": None, "latched": False, "acknowledged": False, "time": None},
        None,
    ]
    assert page["partial"] == []
    assert page["facets"]["alarm"] == [
        {"value": "MAJOR", "count": 1},
        {"value": "none", "count": 2},
    ]
    await directory.aclose()


async def test_a_feed_without_an_answer_marks_no_row_and_says_why(tmp_path: Path) -> None:
    directory = _alarmed_directory(tmp_path)
    FakeAlarmState.held = Unavailable("alarm feed still loading")

    page = await directory.search(words="SR")

    assert all("alarm" not in row for row in page["rows"])
    assert page["partial"] == [{"part": "alarm", "reason": "alarm feed still loading"}]
    assert "alarm" not in page["facets"]
    filtered = await directory.search(words="SR", alarm=["MAJOR"])
    assert filtered["outcome"] == "unavailable"
    assert filtered["reason"] == "cannot filter by alarm: alarm feed still loading"
    await directory.aclose()


async def test_no_alarm_source_adds_no_alarm_key_and_no_partial(tmp_path: Path) -> None:
    FakeDirectory.rows = [_row("SR:PV1")]
    directory = _service(_config(tmp_path, LIVE_BUILD, FACILITY), tmp_path)

    page = await directory.search(words="SR")

    assert "alarm" not in page["rows"][0]
    assert page["partial"] == []
    assert "alarm" not in page["facets"]
    await directory.aclose()


async def test_an_alarm_filter_cuts_the_page_from_the_fetched_set(tmp_path: Path) -> None:
    directory = _alarmed_directory(tmp_path)

    major = await directory.search(words="SR", alarm=["MAJOR"])
    quiet = await directory.search(words="SR", alarm=["none"], page_size=1)

    assert [row["name"] for row in major["rows"]] == ["SR:PV1"]
    assert major["total"] == 1
    assert major["searched"] == "name or description matches SR · alarm MAJOR"
    assert [row["name"] for row in quiet["rows"]] == ["SR:PV2"]
    assert (quiet["total"], quiet["truncated"]) == (2, True)
    assert quiet["facets"]["alarm"] == [
        {"value": "MAJOR", "count": 1},
        {"value": "none", "count": 2},
    ]
    assert quiet["facets"]["origin"] == [{"value": "directory", "count": 2}]
    assert (await directory.count(words="SR", alarm=["none"]))["total"] == 2
    assert (await directory.count(words="SR", alarm=["MINOR"]))["outcome"] == "no_match"
    await directory.aclose()


async def test_an_alarm_filter_above_the_filter_cap_is_unavailable(tmp_path: Path) -> None:
    directory = _alarmed_directory(tmp_path, filter_cap=2)

    page = await directory.search(words="SR", alarm=["MAJOR"])
    counted = await directory.count(words="SR", alarm=["MAJOR"])

    reason = "too many results to filter (3); narrow the search"
    assert (page["outcome"], page["reason"]) == ("unavailable", reason)
    assert (counted["outcome"], counted["reason"]) == ("unavailable", reason)
    await directory.aclose()


# -- the archive join -----------------------------------------------------------

_APPLIANCE = {"type": "epics_archiver", "settings": {"url": "http://archiver.example"}}


@dataclasses.dataclass(frozen=True)
class _Held:
    """A ``store_extent`` answer: the channels the store records, or unknown and why."""

    channels: frozenset[str] | None = None
    oldest: Any = None
    newest: Any = None
    reason: str | None = None


class FakeArchiver:
    """An archiver connector whose store records ``held``."""

    held: Any = _Held(channels=frozenset())

    def __init__(self) -> None:
        self.disconnected = False

    async def store_extent(self) -> Any:
        return self.held

    async def disconnect(self) -> None:
        self.disconnected = True


@pytest.fixture
def archivers(monkeypatch: pytest.MonkeyPatch) -> list[FakeArchiver]:
    """Every archiver connector the service builds, in order."""
    made: list[FakeArchiver] = []

    async def create(config: Any) -> FakeArchiver:
        assert config == _APPLIANCE
        made.append(FakeArchiver())
        return made[-1]

    monkeypatch.setattr(service.ConnectorFactory, "create_archiver_connector", create)
    monkeypatch.setattr(FakeArchiver, "held", _Held(channels=frozenset({"SR:PV1", "SR:PV3"})))
    return made


def _archived_directory(tmp_path: Path, *, feed: bool = False, archive: bool = True) -> Any:
    """Three rows on ``live``, an archive recording PV1 and PV3; PV1 MAJOR with *feed*."""
    FakeDirectory.rows = [_row(f"SR:PV{i}", server="ioc-a") for i in (1, 2, 3)]
    FakeAlarmState.held = {"SR:PV1": _MAJOR}
    section = {**FACILITY, **({"alarm_state": _FEED} if feed else {})}
    config = _config(tmp_path, LIVE_BUILD, section, "SR:PV1", "SR:PV2")
    if archive:
        config["archiver"] = _APPLIANCE
    return _service(config, tmp_path)


async def test_search_marks_each_row_archived_from_the_stores_channels(
    tmp_path: Path, archivers: list[FakeArchiver]
) -> None:
    directory = _archived_directory(tmp_path)

    page = await directory.search(words="SR")

    assert [row["archived"] for row in page["rows"]] == [True, False, True]
    assert "archived_reason" not in page
    assert page["facets"]["archived"] == [
        {"value": "yes", "count": 2},
        {"value": "no", "count": 1},
    ]
    assert len(archivers) == 1
    await directory.aclose()


@pytest.mark.usefixtures("archivers")
async def test_an_archive_that_cannot_list_its_channels_marks_every_row_unknown(
    tmp_path: Path,
) -> None:
    FakeArchiver.held = _Held(reason="this archiver does not report what it holds")
    directory = _archived_directory(tmp_path)

    page = await directory.search(words="SR")
    filtered = await directory.search(words="SR", archived=["no"])
    counted = await directory.count(words="SR", archived=["no"])

    assert [row["archived"] for row in page["rows"]] == [None, None, None]
    assert page["archived_reason"] == "this archiver does not report what it holds"
    assert page["facets"]["archived"] == [{"value": "unknown", "count": 3}]
    reason = "cannot filter by archive: this archiver does not report what it holds"
    assert (filtered["outcome"], filtered["reason"]) == ("unavailable", reason)
    assert (counted["outcome"], counted["reason"]) == ("unavailable", reason)
    await directory.aclose()


@pytest.mark.usefixtures("archivers")
async def test_a_connectors_release_without_store_extent_reads_as_unknown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delattr(FakeArchiver, "store_extent")
    directory = _archived_directory(tmp_path)

    page = await directory.search(words="SR")

    assert [row["archived"] for row in page["rows"]] == [None, None, None]
    assert page["archived_reason"] == (
        "installed osprey-connectors cannot list the archive's channels (upgrade it)"
    )
    await directory.aclose()


async def test_no_archive_on_the_target_adds_no_key_and_refuses_the_filter(
    tmp_path: Path, archivers: list[FakeArchiver]
) -> None:
    directory = _archived_directory(tmp_path, archive=False)

    page = await directory.search(words="SR")
    filtered = await directory.search(words="SR", archived=["yes"])

    assert all("archived" not in row for row in page["rows"])
    assert "archived" not in page["facets"]
    assert filtered["reason"] == "cannot filter by archive: not available for this target"
    assert archivers == []
    await directory.aclose()


@pytest.mark.usefixtures("archivers")
async def test_an_archive_filter_cuts_the_page_and_the_count(tmp_path: Path) -> None:
    directory = _archived_directory(tmp_path)

    page = await directory.search(words="SR", archived=["no"])

    assert [row["name"] for row in page["rows"]] == ["SR:PV2"]
    assert page["total"] == 1
    assert page["searched"] == "name or description matches SR · archived no"
    assert (await directory.count(words="SR", archived=["yes"]))["total"] == 2
    await directory.aclose()


@pytest.mark.usefixtures("archivers")
async def test_each_joined_group_counts_under_the_other_groups_filters(tmp_path: Path) -> None:
    directory = _archived_directory(tmp_path, feed=True)

    by_archive = await directory.search(words="SR", archived=["yes"])
    by_alarm = await directory.search(words="SR", alarm=["MAJOR"])

    # The archive filter narrows the origin and alarm groups, never its own.
    assert by_archive["facets"]["origin"] == [
        {"value": "curated", "count": 1},
        {"value": "directory", "count": 1},
    ]
    assert by_archive["facets"]["alarm"] == [
        {"value": "MAJOR", "count": 1},
        {"value": "none", "count": 1},
    ]
    assert by_archive["facets"]["archived"] == [
        {"value": "yes", "count": 2},
        {"value": "no", "count": 1},
    ]
    assert by_archive["facets"]["ioc"] == [{"value": "ioc-a", "count": 2}]
    # The alarm filter narrows the archive group.
    assert by_alarm["facets"]["archived"] == [{"value": "yes", "count": 1}]
    await directory.aclose()


async def test_a_store_that_cannot_be_reached_is_tried_once_until_asked_fresh(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    attempts: list[Any] = []

    async def unreachable(config: Any) -> Any:
        attempts.append(config)
        raise ConnectionError("connection refused")

    monkeypatch.setattr(service.ConnectorFactory, "create_archiver_connector", unreachable)
    directory = _archived_directory(tmp_path)

    first = await directory.search(words="SR")
    second = await directory.search(words="SR")
    connector, _source, reason = await directory.archive(fresh=True)

    assert first["archived_reason"] == "cannot reach the archive: ConnectionError"
    assert second["archived_reason"] == first["archived_reason"]
    assert connector is None and reason == first["archived_reason"]
    assert len(attempts) == 2
    await directory.aclose()


async def test_a_failed_archive_build_never_serves_the_exception_text(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def leaking(config: Any) -> Any:  # noqa: ARG001 - the factory's signature
        raise ConnectionError("cannot reach mongodb://reader:secret@archive.example:27017")

    monkeypatch.setattr(service.ConnectorFactory, "create_archiver_connector", leaking)
    directory = _archived_directory(tmp_path)

    answer = await directory.search(words="SR")
    _connector, _source, reason = await directory.archive()

    assert reason == answer["archived_reason"] == "cannot reach the archive: ConnectionError"
    assert "mongodb://" not in str(answer) and "secret" not in str(answer)
    await directory.aclose()


async def test_a_store_that_hangs_on_connect_is_waited_for_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    attempts: list[Any] = []

    async def hanging(config: Any) -> Any:
        attempts.append(config)
        await asyncio.sleep(60)

    monkeypatch.setattr(service.ConnectorFactory, "create_archiver_connector", hanging)
    directory = _archived_directory(tmp_path)

    pages = [await directory.search(words="SR") for _ in range(3)]

    assert len(attempts) == 1
    assert pages[-1]["archived_reason"] == "the archive did not answer within 0.3 s"
    await directory.aclose()


@pytest.mark.usefixtures("archivers")
async def test_a_store_whose_extent_read_hangs_is_read_once_until_asked_fresh(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reads: list[None] = []

    async def hanging(_self: FakeArchiver) -> Any:
        reads.append(None)
        await asyncio.sleep(60)

    monkeypatch.setattr(FakeArchiver, "store_extent", hanging)
    directory = _archived_directory(tmp_path)

    pages = [await directory.search(words="SR") for _ in range(3)]
    assert len(reads) == 1
    assert pages[-1]["archived_reason"] == "no answer within 0.3 s"

    await directory.archive_extent(fresh=True)
    assert len(reads) == 2
    await directory.aclose()


async def test_the_archive_connector_is_disconnected_when_the_target_moves(
    tmp_path: Path, archivers: list[FakeArchiver], record: _Record
) -> None:
    record.move("live", 1)
    directory = _archived_directory(tmp_path)

    await directory.search(words="SR")
    first = archivers[0]
    connector, source, reason = await directory.archive()
    assert connector is first and reason is None
    assert _connectors.as_payload(source) == {
        "target": "live",
        "service": "epics_archiver",
        "label": "epics_archiver",
        "rehearsal": False,
    }

    record.move("live", 2)
    await directory.search(words="SR")

    assert first.disconnected is True
    assert len(archivers) == 2
    await directory.aclose()
    assert archivers[1].disconnected is True


# -- describe -----------------------------------------------------------------


async def test_describe_marks_the_record_and_removes_hidden_properties(tmp_path: Path) -> None:
    FakeDirectory.records = {
        "SR:BPM1": DirectoryRecord(
            row=_row("SR:BPM1", status="Inactive", host="h1"),
            properties={"Engineer": "A. Person", "secret": "x"},
        )
    }
    section = {**FACILITY, "display": {"hidden_properties": ["secret", "hostName"]}}
    directory = _service(_config(tmp_path, LIVE_BUILD, section, "SR:BPM1"), tmp_path)

    found = await directory.describe("SR:BPM1")
    missing = await directory.describe("SR:NOPE")

    assert found["outcome"] == "inactive"
    assert found["origin"] == "curated"
    assert found["record"]["row"]["origin"] == "curated"
    assert found["record"]["row"]["host"] is None
    assert found["record"]["properties"] == {"Engineer": "A. Person"}
    assert missing["outcome"] == "not_listed"
    assert missing["origin"] == "directory"
    assert missing["record"] is None
    await directory.aclose()


def _described_rows(name: str) -> None:
    FakeDirectory.rows = [_row(name, description="Cavity temperature", server="ioc-a")]
    FakeDirectory.records = {
        name: DirectoryRecord(row=_row(name, description="Cavity temperature"))
    }


async def test_hiding_the_configured_description_property_clears_the_description(
    tmp_path: Path,
) -> None:
    _described_rows("SR:BPM1")
    hidden = {**FACILITY, "description_property": "Comment"}
    directory = _service(
        _config(tmp_path, LIVE_BUILD, {**hidden, "display": {"hidden_properties": ["Comment"]}}),
        tmp_path,
    )

    page = await directory.search(words="SR")
    found = await directory.describe("SR:BPM1")

    assert [row["description"] for row in page["rows"]] == [None]
    assert found["record"]["row"]["description"] is None
    await directory.aclose()

    directory = _service(
        _config(tmp_path, LIVE_BUILD, {**hidden, "display": {"hidden_properties": ["recordDesc"]}}),
        tmp_path,
    )

    page = await directory.search(words="SR")
    found = await directory.describe("SR:BPM1")

    assert [row["description"] for row in page["rows"]] == ["Cavity temperature"]
    assert found["record"]["row"]["description"] == "Cavity temperature"
    await directory.aclose()


async def test_hiding_reads_the_description_property_the_backend_was_built_with(
    tmp_path: Path,
) -> None:
    _described_rows("SR:BPM1")
    section = {
        **FACILITY,
        "description_property": "Comment",
        "display": {"hidden_properties": ["Comment"]},
    }
    directory = _service(_config(tmp_path, VA_BUILD, section), tmp_path)

    page = await directory.search(words="SR")
    found = await directory.describe("SR:BPM1")
    backend, source, _reason = await directory.backend("pv_directory")

    assert (backend.spec.type, source.target) == ("simulator_view", "va")
    assert backend.spec.settings["description_property"] == "Comment"
    assert [row["description"] for row in page["rows"]] == [None]
    assert found["record"]["row"]["description"] is None
    await directory.aclose()


# -- targets ------------------------------------------------------------------


async def test_backend_per_target(tmp_path: Path, record: _Record) -> None:
    section = {
        **FACILITY,
        "alarm_log": {"type": "phoebus_alarm_logger", "settings": {"url": "http://al"}},
        "alarm_state": {"replay_file": "demo:example-alarms"},
    }
    directory = _service(_config(tmp_path, VA_BUILD, section), tmp_path)

    backend, source, reason = await directory.backend("pv_directory")
    assert (backend.spec.type, source.target, reason) == ("simulator_view", "va", None)
    assert await directory.backend("alarm_log") == (None, None, _connectors.NO_SOURCE_REASON)

    record.move("standin", 1)
    backend, source, _reason = await directory.backend("alarm_log")
    assert (backend.spec.type, source.target, source.rehearsal) == (
        "phoebus_alarm_logger",
        "standin",
        True,
    )
    assert await directory.backend("put_log") == (None, None, _connectors.NO_SOURCE_REASON)
    await directory.aclose()


async def test_backend_carries_the_build_failure_reason(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, record: _Record
) -> None:
    async def broken(*_: Any, **__: Any) -> Any:
        raise RuntimeError("no route to host")

    patch_create_service(monkeypatch, "alarm_log", broken)
    section = {
        **FACILITY,
        "alarm_log": {"type": "phoebus_alarm_logger", "settings": {"url": "http://al"}},
    }
    record.move("live", 1)
    directory = _service(_config(tmp_path, LIVE_BUILD, section), tmp_path)

    assert await directory.backend("alarm_log") == (
        None,
        None,
        "cannot build the alarm_log backend 'phoebus_alarm_logger': no route to host",
    )
    await directory.aclose()


async def test_va_never_reaches_a_facility_backend(tmp_path: Path, built: list[Any]) -> None:
    section = {
        **FACILITY,
        "alarm_log": {"type": "phoebus_alarm_logger", "settings": {"url": "http://al"}},
        "put_log": {"type": "caputlog", "settings": {"url": "http://pl"}},
        "alarm_state": {
            "type": "phoebus_alarm_server",
            "settings": {"bootstrap_servers": "k:9092", "topic": "Accelerator"},
            "replay_file": "demo:example-alarms",
        },
    }
    directory = _service(_config(tmp_path, VA_BUILD, section), tmp_path)

    await directory.search(words="SR")
    await directory.describe("SR:BPM1")
    await directory.count()
    await directory.snapshot()
    await directory.probes()
    for kind in _FAKES:
        await directory.backend(kind)

    assert {(b.spec.kind, b.spec.type) for b in built} == {
        ("directory", "simulator_view"),
        ("alarm_state", "alarm_replay"),
    }
    assert {b.spec.target for b in built} == {"va"}
    await directory.aclose()


async def test_va_to_live_never_answers_from_the_replay(tmp_path: Path, record: _Record) -> None:
    section = {**FACILITY, "alarm_state": {"replay_file": "demo:example-alarms"}}
    directory = _service(_config(tmp_path, VA_BUILD, section), tmp_path)

    on_va = await directory.snapshot()
    assert on_va is not None
    assert on_va.target == "va"

    record.move("live", 1)

    assert await directory.snapshot() is None
    assert await directory.backend("alarm_state") == (
        None,
        None,
        _connectors.NO_SOURCE_REASON,
    )
    await directory.aclose()


async def test_demo_replay_is_rewritten_to_the_packaged_file(tmp_path: Path) -> None:
    section = {"alarm_state": {"replay_file": "demo:example-alarms"}}
    directory = _service(_config(tmp_path, VA_BUILD, section), tmp_path)

    backend, _source, _reason = await directory.backend("alarm_state")

    expected = resolve_replay_file("demo:example-alarms", tmp_path)
    assert backend.spec.settings["replay_file"] == str(expected)
    assert Path(backend.spec.settings["replay_file"]).is_absolute()
    await directory.aclose()


async def test_snapshot_is_one_per_target_and_generation(tmp_path: Path, record: _Record) -> None:
    section = {
        **FACILITY,
        "alarm_state": {
            "type": "phoebus_alarm_server",
            "settings": {"bootstrap_servers": "k:9092", "topic": "Accelerator"},
        },
    }
    directory = _service(_config(tmp_path, LIVE_BUILD, section), tmp_path)

    first = await directory.snapshot()
    assert await directory.snapshot() is first
    assert (await first.state_for("SR:BPM1", 0.5)).severity == "MAJOR"

    record.move("live", 1)
    second = await directory.snapshot()

    assert second is not first
    assert second.target == "live"
    assert first._task is None  # closed
    await directory.aclose()
    assert second._task is None


async def test_target_move_clears_the_cache(
    tmp_path: Path, built: list[Any], record: _Record
) -> None:
    FakeDirectory.rows = [_row("SR:BPM1")]
    record.move("live", 1)
    directory = _service(_config(tmp_path, LIVE_BUILD, FACILITY), tmp_path)

    await directory.search(words="SR")
    await directory.search(words="SR", page=1)
    assert len(built) == 1
    assert len(built[0].counts) == 1

    record.move("live", 2)
    await directory.search(words="SR")

    assert len(built) == 2
    assert len(built[1].counts) == 1
    await directory.aclose()


async def test_no_source_form_at_standin(tmp_path: Path, built: list[Any], record: _Record) -> None:
    record.move("standin", 1)
    section = {"alarm_state": {"replay_file": "demo:example-alarms"}}
    directory = _service(_config(tmp_path, VA_BUILD, section), tmp_path)

    expected = _connectors.unavailable_payload(_connectors.NO_SOURCE_REASON, target="standin")

    assert await directory.search(words="SR") == _no_source_page(expected)
    assert await directory.count() == {**expected, "total": None}
    described = await directory.describe("SR:BPM1")
    assert {key: described[key] for key in expected} == expected
    assert expected["source"]["rehearsal"] is True
    assert await directory.snapshot() is None
    assert built == []
    await directory.aclose()


def _no_source_page(form: dict[str, Any]) -> dict[str, Any]:
    """The whole ``DirectoryPage`` a search answers around the No-source *form*."""
    return {
        **form,
        "searched": "",
        "request": "",
        "total": None,
        "rows": [],
        "truncated": False,
        "partial": [],
        "facets": None,
        "facets_reason": None,
    }


async def test_a_search_with_no_source_on_live_is_the_shared_fixture_page(
    tmp_path: Path, record: _Record
) -> None:
    record.move("live", 1)
    section = {"alarm_state": {"replay_file": "demo:example-alarms"}}
    directory = _service(_config(tmp_path, VA_BUILD, section), tmp_path)

    shared = json.loads(_ROUTES.read_text(encoding="utf-8"))["search no source"]

    assert await directory.search(words="SR") == shared
    await directory.aclose()


async def test_every_search_outcome_carries_the_page_fields(
    tmp_path: Path, record: _Record
) -> None:
    page_keys = {field.name for field in dataclasses.fields(DirectoryPage)}
    FakeDirectory.rows = [_row("SR:BPM1")]
    record.move("live", 1)
    found = await _service(_config(tmp_path, LIVE_BUILD, FACILITY), tmp_path).search(words="SR")
    FakeDirectory.search_delay = 1.0
    late = await _service(_config(tmp_path, LIVE_BUILD, FACILITY), tmp_path).search(words="SR")
    no_source = await _service(_config(tmp_path, LIVE_BUILD, {"display": {}}), tmp_path).search(
        words="SR"
    )

    assert found["outcome"] == "found"
    assert (late["outcome"], late["reason"]) == ("unavailable", "no answer within 0.3 s")
    assert (no_source["outcome"], no_source["reason"]) == (
        "unavailable",
        _connectors.NO_SOURCE_REASON,
    )
    for answer in (found, late, no_source):
        assert set(answer) == page_keys, answer["outcome"]


@pytest.mark.parametrize(
    ("control_system", "target", "service_type"),
    [(VA_BUILD, "va", "simulator_view"), (LIVE_BUILD, "live", "channelfinder")],
)
async def test_no_record_answers_from_the_baseline(
    tmp_path: Path,
    built: list[Any],
    control_system: dict[str, Any],
    target: str,
    service_type: str,
) -> None:
    directory = _service(_config(tmp_path, control_system, FACILITY), tmp_path)

    page = await directory.search(words="SR")

    assert page["source"]["target"] == target
    assert page["source"]["service"] == service_type
    assert built[0].project_dir == tmp_path
    await directory.aclose()


# -- probes and lifetime ------------------------------------------------------


async def test_probes_name_every_configured_service(tmp_path: Path, record: _Record) -> None:
    section = {
        **FACILITY,
        "alarm_log": {"type": "phoebus_alarm_logger", "settings": {"url": "http://al"}},
    }
    directory = _service(_config(tmp_path, LIVE_BUILD, section), tmp_path)

    probes = await directory.probes()

    assert set(probes) == {"pv_directory", "alarm_log"}
    assert probes["pv_directory"]["outcome"] == "found"
    assert probes["pv_directory"]["evidence"] == "channelfinder answered"

    record.move("va", 1)
    on_va = await directory.probes()
    assert on_va["alarm_log"] == _connectors.unavailable_payload(
        _connectors.NO_SOURCE_REASON, target="va"
    )
    await directory.aclose()


_KAFKA = {
    "type": "phoebus_alarm_server",
    "settings": {"bootstrap_servers": "k:9092", "topic": "Accelerator"},
}


async def test_the_alarm_feed_probe_starts_the_feed_through_the_snapshot(tmp_path: Path) -> None:
    directory = _service(
        _config(tmp_path, LIVE_BUILD, {**FACILITY, "alarm_state": _KAFKA}), tmp_path
    )

    probes = await directory.probes(kinds=("alarm_state",))

    assert set(probes) == {"alarm_state"}
    assert probes["alarm_state"]["outcome"] == "found"
    assert probes["alarm_state"]["evidence"] == "phoebus_alarm_server answered"
    snapshot = await directory.snapshot()
    assert snapshot is not None and snapshot._task is not None
    await directory.aclose()
    assert snapshot._task is None


async def test_probes_of_other_kinds_start_no_feed(tmp_path: Path) -> None:
    directory = _service(
        _config(tmp_path, LIVE_BUILD, {**FACILITY, "alarm_state": _KAFKA}), tmp_path
    )

    probes = await directory.probes(kinds=("pv_directory", "alarm_log", "put_log"))

    assert set(probes) == {"pv_directory"}
    assert directory._snapshot is None
    await directory.aclose()


async def test_an_alarm_feed_that_cannot_be_built_carries_its_reason(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def refuse(*_: Any, **__: Any) -> Any:
        raise RuntimeError("no broker setting")

    patch_create_service(monkeypatch, "alarm_state", refuse)
    directory = _service(
        _config(tmp_path, LIVE_BUILD, {**FACILITY, "alarm_state": _KAFKA}), tmp_path
    )

    probes = await directory.probes(kinds=("alarm_state",))

    assert probes["alarm_state"]["outcome"] == "unavailable"
    assert probes["alarm_state"]["reason"] == (
        "cannot build the alarm_state backend 'phoebus_alarm_server': no broker setting"
    )
    await directory.aclose()


async def test_aclose_leaves_no_task(tmp_path: Path) -> None:
    section = {"alarm_state": {"replay_file": "demo:example-alarms"}}
    directory = _service(_config(tmp_path, VA_BUILD, section), tmp_path)
    snapshot = await directory.snapshot()
    await snapshot.state_for("X", 0.2)

    await directory.aclose()
    await directory.aclose()

    others = {t for t in asyncio.all_tasks() if t is not asyncio.current_task()}
    assert not [t for t in others if t.get_name().startswith("alarm-state-")]
    assert snapshot._task is None


# -- the connectors floor -----------------------------------------------------


@pytest.fixture
def floor() -> Iterator[ModuleType]:
    """The service re-imported over a shim whose PV directory modules are unimportable."""
    with pytest.MonkeyPatch.context() as patch:
        for name in _NEW_MODULES:
            patch.setitem(sys.modules, name, None)
        importlib.reload(_connectors)
        yield importlib.reload(service)
    importlib.reload(_connectors)
    importlib.reload(service)


async def test_modules_blocked_every_read_answers_the_floor(
    tmp_path: Path, floor: ModuleType
) -> None:
    directory = floor.DirectoryService(
        _config(tmp_path, LIVE_BUILD, FACILITY), project_dir=tmp_path
    )
    expected = _connectors.unavailable_payload(_connectors.FLOOR_REASON, target="live")

    assert await directory.search(words="SR") == _no_source_page(expected)
    assert await directory.count() == {**expected, "total": None}
    described = await directory.describe("SR:BPM1")
    assert {key: described[key] for key in expected} == expected
    assert await directory.backend("pv_directory") == (None, None, _connectors.FLOOR_REASON)
    assert await directory.snapshot() is None
    assert await directory.probes() == {"pv_directory": expected}
    await directory.aclose()


def test_modules_blocked_the_framework_takes_only_listed_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tests.connectors import test_public_api

    for name in _NEW_MODULES:
        monkeypatch.setitem(sys.modules, name, None)
    test_public_api._framework_imports.cache_clear()
    try:
        taken = [
            (module, name)
            for where, module, name in test_public_api._framework_imports()
            if where.startswith("src/osprey/services/pv_directory/service.py")
        ]
        test_public_api.test_the_framework_takes_only_listed_names()
    finally:
        test_public_api._framework_imports.cache_clear()

    assert taken
    assert all(
        not any(module == new or module.startswith(new + ".") for new in _NEW_MODULES)
        for module, _ in taken
    )
