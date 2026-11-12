"""Every directory-kind backend the factory builds answers the one contract.

Each case is built through ``ConnectorFactory.create_<kind>_connector`` with both
seams passed, as the framework passes them to every backend of a kind: a recording
transport for the HTTP backends, a consumer that offers only reads for the Kafka
feed, a replay file on disk, a simulator view under the project directory, and an
out-of-tree class selected by its dotted path.

The recording transport fails the test on any request that is not a read or a
search, and the loopback service behind the login-page cases records what reached
the wire, so both halves of "every request is a read or a search" are checked.
"""

from __future__ import annotations

import asyncio
import json
import sys
import threading
import types
from collections import namedtuple
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import pytest

from osprey_connectors._service_http import LOGIN_PAGE_REASON, ReadOnlyTransport
from osprey_connectors.alarm_state import AlarmConfigRecord, AlarmStateRecord, _phoebus_kafka
from osprey_connectors.directory import (
    ChannelDirectory,
    DirectoryPage,
    DirectoryQuery,
    DirectoryRecord,
    DirectoryRow,
)
from osprey_connectors.factory import ConnectorFactory
from osprey_connectors.service_answers import BackendSpec as Spec
from osprey_connectors.service_answers import HistoryPage, Outcome, Unavailable
from osprey_connectors.service_targets import service_targets
from osprey_connectors_tests._harness import isolated_connector_registries
from osprey_connectors_tests.directory.test_simulator_directory import _write_view

DEADLINE = 6.0
PV = "DEMO:RF1:CAV:Temp-I"
SIMULATED_PV = "T:SR:QF1:SP"
ABSENT_PV = "DEMO:NO:SUCH:PV"
NO_ANSWER = "no answer within 6 s"
URL = "https://directory.example.org"
TARGETS = ("va", "live", "standin")

CF_FIXTURES = Path(__file__).parent / "fixtures" / "channelfinder"
REPLAY_FILE = Path(__file__).parents[1] / "alarm_state" / "fixtures" / "old-feed.ndjson"

CF_CHANNELS = "/resources/channels"
CF_COUNT = "/resources/channels/count"
CF_PROPERTIES = "/resources/properties"

Record = namedtuple("Record", "key value")
Answer = Callable[[str, str, Any], Any]


# --- the out-of-tree backend ------------------------------------------------------

_MEMORY_RECORD = DirectoryRecord(
    row=DirectoryRow(name=PV, status="Active", description="Cavity 1 temperature", ioc="rf1-cav"),
    properties={"iocName": "rf1-cav"},
)


class MemoryDirectory(ChannelDirectory):
    """An out-of-tree directory, selected by its dotted path, holding one PV in memory."""

    def _hits(self, query: DirectoryQuery) -> list[DirectoryRecord]:
        row = _MEMORY_RECORD.row
        text = f"{row.name} {row.description}".casefold()
        matches = all(word.casefold() in text for word in query.words.split()) and (
            not query.name or query.name.removeprefix("=") in row.name
        )
        return [_MEMORY_RECORD] if matches else []

    async def search(self, query: DirectoryQuery, deadline: float) -> DirectoryPage:  # noqa: ARG002
        hits = self._hits(query)
        return DirectoryPage(
            outcome=Outcome.FOUND if hits else Outcome.NO_MATCH,
            searched=query.words or "every PV",
            request="memory",
            source=self.source,
            total=len(hits),
            rows=[hit.row for hit in hits],
        )

    async def count(self, query: DirectoryQuery, deadline: float) -> int:  # noqa: ARG002
        return len(self._hits(query))

    async def describe(self, name: str, deadline: float) -> Any:  # noqa: ARG002
        return _MEMORY_RECORD if name == _MEMORY_RECORD.row.name else Outcome.NOT_LISTED

    async def property_names(self, deadline: float) -> list[str]:  # noqa: ARG002
        return sorted(_MEMORY_RECORD.properties)

    async def property_values(self, name: str, prefix: str | None, deadline: float) -> list[str]:  # noqa: ARG002
        value = _MEMORY_RECORD.properties.get(name)
        return [value] if value is not None and value.startswith(prefix or "") else []

    async def probe(self, deadline: float) -> str:  # noqa: ARG002
        return "memory answered: 1 PV"


# --- seams ------------------------------------------------------------------------


def _is_read_or_search(method: str, path: str, json_body: Any) -> bool:
    """A GET with no body, or a POST of a query to a search route."""
    if method == "GET":
        return json_body is None
    return method == "POST" and path.rsplit("/", 1)[-1] == "search"


@dataclass(frozen=True)
class Sent:
    method: str
    path: str
    params: list[tuple[str, Any]]
    json_body: Any
    deadline: float


class RecordingTransport:
    """Answers by path like a service would; fails the test on any request that writes."""

    def __init__(self, answer: Answer) -> None:
        self._answer = answer
        self.sent: list[Sent] = []
        self.violations: list[str] = []

    async def request(
        self,
        method: str,
        path: str,
        *,
        params: Any = None,
        json_body: Any = None,
        deadline: float,
    ) -> Any:
        if not _is_read_or_search(method, path, json_body):
            self.violations.append(f"{method} {path}")
            pytest.fail(f"{method} {path} is neither a read nor a search")
        pairs = list(params.items() if isinstance(params, dict) else params or [])
        self.sent.append(Sent(method, path, pairs, json_body, deadline))
        if deadline <= 0:
            return Unavailable(f"no answer within {deadline:g} s")
        return self._answer(method, path, params)


class ReadOnlyConsumer:
    """aiokafka's read surface over one partition; reaching for anything else fails the test."""

    def __init__(self, records: list[Record] = (), *, start_error: Exception | None = None):  # type: ignore[assignment]
        self.violations: list[str] = []
        self.records = list(records)
        self.start_error = start_error
        self.position_at = 0
        self.stopped = False

    def __getattr__(self, name: str) -> Any:
        if name.startswith("__"):
            raise AttributeError(name)
        self.violations.append(name)
        pytest.fail(f"the alarm feed reached for consumer.{name}: only reads are allowed")

    async def start(self) -> None:
        if self.start_error is not None:
            raise self.start_error

    async def stop(self) -> None:
        self.stopped = True

    def assignment(self) -> set[str]:
        return {"topic-0"}

    async def end_offsets(self, partitions: Any) -> dict[str, int]:
        return dict.fromkeys(partitions, len(self.records))

    async def seek_to_beginning(self, *_partitions: Any) -> None:
        self.position_at = 0

    async def position(self, _partition: Any) -> int:
        return self.position_at

    async def getmany(self, *_partitions: Any, **_options: Any) -> dict[str, list[Record]]:
        await asyncio.sleep(0)
        batch = self.records[self.position_at :]
        self.position_at += len(batch)
        return {"topic-0": batch} if batch else {}


# --- recorded answers -------------------------------------------------------------


def _fixture(name: str) -> Any:
    return json.loads((CF_FIXTURES / name).read_text(encoding="utf-8"))


_TREE_PATH = f"/Demo/RF/{PV}"
_LOGGED_ROWS = [
    {
        "severity": "MAJOR",
        "message": "HIHI_ALARM",
        "time": "2026-10-29T03:14:07.120Z",
        "current_message": "NO_ALARM",
        "config": f"state:{_TREE_PATH}",
        "pv": PV,
    },
    {
        "message_time": "2026-10-29T03:13:00.000Z",
        "config": f"config:{_TREE_PATH}",
        "config_msg": json.dumps({"enabled": True, "latching": True, "guidance": []}),
    },
]
_PUT_HIT = {
    "pv": {"raw": f"{PV}.HIHI"},
    "old": {"raw": "45"},
    "new": {"raw": "50"},
    "ioctimestamp": {"raw": "2026-10-28T21:03:11.000Z"},
}


def _by_path(answers: dict[str, Any], default: Any) -> Answer:
    return lambda _method, path, _params: answers.get(path, default)


def _always(value: Any) -> Answer:
    return lambda _method, _path, _params: value


_MALFORMED = {
    "malformed-null": None,
    "malformed-text": "<not json>",
    "malformed-mapping": {"unexpected": True},
    "malformed-list": [1, "x", None],
    "malformed-number": 7,
}


def _http_answers(recorded: Answer, empty: Answer) -> dict[str, Answer]:
    return {
        "recorded": recorded,
        "empty": empty,
        "unavailable": _always(Unavailable(NO_ANSWER)),
        **{mode: _always(value) for mode, value in _MALFORMED.items()},
    }


_HTTP_ANSWERS: dict[str, dict[str, Answer]] = {
    "channelfinder": _http_answers(
        _by_path(
            {
                CF_CHANNELS: _fixture("channels.json"),
                CF_COUNT: _fixture("count.json"),
                CF_PROPERTIES: _fixture("properties.json"),
            },
            [],
        ),
        _by_path({CF_COUNT: 0}, []),
    ),
    "phoebus_alarm_logger": _http_answers(
        _by_path(
            {"/": {"name": "alarm-logger", "version": "4.7.3"}, "/search/alarm": _LOGGED_ROWS}, []
        ),
        _by_path({"/": {}}, []),
    ),
    "caputlog": _http_answers(
        _always({"results": [_PUT_HIT], "totalResults": 1}),
        _always({"results": [], "totalResults": 0}),
    ),
}

_FEED = [
    Record(f"config:{_TREE_PATH}".encode(), b'{"enabled": true}'),
    Record(f"state:{_TREE_PATH}".encode(), b'{"severity": "MAJOR", "current_severity": "MAJOR"}'),
]


# --- the cases --------------------------------------------------------------------


@dataclass(frozen=True)
class Case:
    """One backend the factory builds, and the conditions it is built under."""

    id: str
    kind: str
    type: str
    pv: str
    modes: tuple[str, ...]

    @property
    def http(self) -> bool:
        return self.id in _HTTP_ANSWERS


@dataclass
class Built:
    backend: Any
    spec: Spec
    transport: RecordingTransport | None = None
    consumer: ReadOnlyConsumer | None = None
    notes: dict[str, Any] = field(default_factory=dict)

    def assert_read_only(self) -> None:
        if self.transport is not None:
            assert self.transport.violations == []
            allowed = ReadOnlyTransport(
                {"url": URL}, type(self.backend).ALLOWED_REQUESTS, where="settings"
            )
            assert all(allowed._allows(sent.method, sent.path) for sent in self.transport.sent)
        if self.consumer is not None:
            assert self.consumer.violations == []


_HTTP_MODES = tuple(_HTTP_ANSWERS["channelfinder"])

CASES = (
    Case("channelfinder", "directory", "channelfinder", PV, _HTTP_MODES),
    Case(
        "simulator_view",
        "directory",
        "simulator_view",
        SIMULATED_PV,
        ("view", "no-view", "no-project"),
    ),
    Case("dotted", "directory", f"{__name__}.MemoryDirectory", PV, ("memory",)),
    Case("phoebus_alarm_logger", "alarm_log", "phoebus_alarm_logger", PV, _HTTP_MODES),
    Case(
        "phoebus_alarm_server",
        "alarm_state",
        "phoebus_alarm_server",
        PV,
        ("feed", "empty-topic", "broker-down", "no-settings", "no-client"),
    ),
    Case(
        "alarm_replay",
        "alarm_state",
        "alarm_replay",
        PV,
        ("replay", "relative", "missing", "unreadable", "unresolved-demo", "unset"),
    ),
    Case("caputlog", "put_log", "caputlog", PV, _HTTP_MODES),
)
HTTP_CASES = tuple(case for case in CASES if case.http)


def _create(kind: str) -> Any:
    """The factory's create method for *kind*; the directory kind's keeps its old stem."""
    stem = "pv_directory" if kind == "directory" else kind
    return getattr(ConnectorFactory, f"create_{stem}_connector")


DIRECTORY_CASES = tuple(case for case in CASES if case.kind == "directory")
HISTORY_CASES = tuple(case for case in CASES if case.kind in ("alarm_log", "put_log"))
ALARM_STATE_CASES = tuple(case for case in CASES if case.kind == "alarm_state")
EVERY_MODE = [
    pytest.param(case, mode, id=f"{case.id}-{mode}") for case in CASES for mode in case.modes
]


def _ids(cases: tuple[Case, ...]) -> list[str]:
    return [case.id for case in cases]


@pytest.fixture(autouse=True)
def _isolated(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setattr(_phoebus_kafka, "_POLL_TIMEOUT_MS", 0)
    monkeypatch.setattr(_phoebus_kafka, "_RETRY_S", 0.0)
    with isolated_connector_registries():
        yield


async def _build(
    case: Case,
    mode: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    target: str = "live",
) -> Built:
    settings: dict[str, Any] = {}
    project_dir: Path | None = tmp_path
    transport = consumer = None
    consumer_factory = None
    if case.http:
        settings = {"url": URL}
        transport = RecordingTransport(_HTTP_ANSWERS[case.id][mode])
    elif case.id == "simulator_view":
        if mode == "view":
            _write_view(tmp_path)
        elif mode == "no-project":
            project_dir = None
    elif case.id == "phoebus_alarm_server":
        if mode in ("feed", "empty-topic", "broker-down"):
            consumer = ReadOnlyConsumer(
                _FEED if mode == "feed" else [],
                start_error=ConnectionError("broker down") if mode == "broker-down" else None,
            )
            consumer_factory = lambda _settings: consumer  # noqa: E731
        elif mode == "no-client":
            settings = {"bootstrap_servers": "broker.demo.lab:9092", "topic": "Demo"}
            monkeypatch.setitem(sys.modules, "aiokafka", None)
    elif case.id == "alarm_replay":
        if mode == "relative":
            (tmp_path / "feed.ndjson").write_bytes(REPLAY_FILE.read_bytes())
        if mode == "unreadable":
            (tmp_path / "feed.ndjson").write_text(
                '{"key": "state:/x"}\nnot json\n', encoding="utf-8"
            )
        replay = {
            "replay": str(REPLAY_FILE),
            "relative": "feed.ndjson",
            "missing": str(tmp_path / "absent.ndjson"),
            "unreadable": str(tmp_path / "feed.ndjson"),
            "unresolved-demo": "demo:alarm-feed",
        }.get(mode)
        if replay is not None:
            settings = {"replay_file": replay}
    spec = Spec(
        kind=case.kind,  # type: ignore[arg-type]
        type=case.type,
        target=target,
        settings=settings,
        rehearsal=target == "standin",
    )
    create = _create(case.kind)
    backend = await create(
        spec, project_dir=project_dir, transport=transport, consumer_factory=consumer_factory
    )
    return Built(backend, spec, transport, consumer)


async def _exercise(backend: Any, case: Case, deadline: float = DEADLINE) -> list[tuple[str, Any]]:
    """Every method of the case's kind, once each, with ordinary arguments."""
    pv = case.pv
    since = datetime.now(UTC) - timedelta(days=30)
    if case.kind == "directory":
        return [
            ("search", await backend.search(DirectoryQuery(words="cooling"), deadline)),
            (
                "search",
                await backend.search(DirectoryQuery(name=pv, page=1, page_size=5), deadline),
            ),
            ("count", await backend.count(DirectoryQuery(words="cooling"), deadline)),
            ("describe", await backend.describe(pv, deadline)),
            ("property_names", await backend.property_names(deadline)),
            ("property_values", await backend.property_values("iocName", "rf", deadline)),
            ("probe", await backend.probe(deadline)),
        ]
    if case.kind in ("alarm_log", "put_log"):
        answers = [
            ("history", await backend.history(pv, None, 10, deadline)),
            ("history", await backend.history(pv, since, 1, deadline)),
        ]
        if case.kind == "alarm_log":
            answers.append(("config_for", await backend.config_for(pv, deadline)))
        return [*answers, ("probe", await backend.probe(deadline))]
    await backend.start()
    await backend.start()
    try:
        return [
            ("state_for", await backend.state_for(pv, deadline)),
            ("state_for", await backend.state_for(ABSENT_PV, deadline)),
            ("config_for", await backend.config_for(pv, deadline)),
            ("probe", await backend.probe(deadline)),
        ]
    finally:
        await backend.aclose()
        await backend.aclose()


_RETURNS: dict[str, tuple[type, ...]] = {
    "search": (DirectoryPage,),
    "count": (int, Unavailable),
    "describe": (DirectoryRecord, Outcome, Unavailable),
    "property_names": (list, Unavailable),
    "property_values": (list, Unavailable),
    "probe": (str, Unavailable),
    "history": (HistoryPage,),
    "config_for": (AlarmConfigRecord, Unavailable),
    "state_for": (AlarmStateRecord, Unavailable, type(None)),
}


def _pages(answers: list[tuple[str, Any]]) -> list[DirectoryPage | HistoryPage]:
    return [answer for _, answer in answers if isinstance(answer, DirectoryPage | HistoryPage)]


# --- the contract -----------------------------------------------------------------


@pytest.mark.parametrize(("case", "mode"), EVERY_MODE)
async def test_every_method_answers_its_declared_type_and_never_raises(
    case, mode, tmp_path, monkeypatch
):
    built = await _build(case, mode, tmp_path, monkeypatch)

    answers = await _exercise(built.backend, case)

    for method, answer in answers:
        assert isinstance(answer, _RETURNS[method]), (method, answer)
        assert not isinstance(answer, bool), method
        if isinstance(answer, Outcome):
            assert answer is Outcome.NOT_LISTED
        if isinstance(answer, Unavailable):
            assert answer.reason
    for page in _pages(answers):
        assert page.outcome is not Outcome.NOT_LISTED
    built.assert_read_only()


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize("case", CASES, ids=_ids(CASES))
async def test_the_source_names_the_specs_kind_target_and_rehearsal(
    case, target, tmp_path, monkeypatch
):
    built = await _build(case, case.modes[0], tmp_path, monkeypatch, target=target)

    source = built.backend.source
    answers = await _exercise(built.backend, case)

    assert source.target == target
    assert source.rehearsal is (target == "standin")
    assert source.service == case.type
    assert source.kind == case.kind
    assert all(page.source == source for page in _pages(answers))


async def test_an_out_of_tree_directory_names_its_source_as_any_directory(tmp_path, monkeypatch):
    built = await _build(CASES[2], "memory", tmp_path, monkeypatch)

    assert isinstance(built.backend, MemoryDirectory)
    assert built.backend.source.kind == "directory"
    assert not hasattr(built.backend.source, "label")


@pytest.mark.parametrize("case", DIRECTORY_CASES, ids=_ids(DIRECTORY_CASES))
async def test_an_exact_name_not_held_is_not_listed_and_a_search_without_rows_is_no_match(
    case, tmp_path, monkeypatch
):
    mode = "empty" if case.http else case.modes[0]
    built = await _build(case, mode, tmp_path, monkeypatch)

    described = await built.backend.describe(ABSENT_PV, DEADLINE)
    page = await built.backend.search(DirectoryQuery(words="nosuchword"), DEADLINE)
    by_name = await built.backend.search(DirectoryQuery(name=f"={ABSENT_PV}"), DEADLINE)

    assert described is Outcome.NOT_LISTED
    assert (page.outcome, page.rows) == (Outcome.NO_MATCH, [])
    assert (by_name.outcome, by_name.rows) == (Outcome.NO_MATCH, [])


@pytest.mark.parametrize("case", HISTORY_CASES, ids=_ids(HISTORY_CASES))
async def test_a_history_without_rows_is_no_match_never_not_listed(case, tmp_path, monkeypatch):
    built = await _build(case, "empty", tmp_path, monkeypatch)

    page = await built.backend.history(case.pv, None, 10, DEADLINE)

    assert (page.outcome, list(page.rows)) == (Outcome.NO_MATCH, [])


@pytest.mark.parametrize(
    "case", [c for c in HTTP_CASES if c.kind == "directory"], ids=lambda c: c.id
)
async def test_a_page_that_arrives_while_the_count_times_out_is_a_partial_page(
    case, tmp_path, monkeypatch
):
    channels = _fixture("channels.json")
    built = await _build(case, "recorded", tmp_path, monkeypatch)
    built.transport._answer = _by_path(
        {CF_CHANNELS: channels, CF_COUNT: Unavailable(NO_ANSWER)}, []
    )

    page = await built.backend.search(DirectoryQuery(words="cooling"), DEADLINE)

    assert page.outcome is Outcome.FOUND
    assert [row.name for row in page.rows] == [channel["name"] for channel in channels]
    assert page.total is None
    assert page.partial == [{"part": "count", "reason": NO_ANSWER}]


@pytest.mark.parametrize("case", HTTP_CASES, ids=_ids(HTTP_CASES))
async def test_no_request_is_given_more_than_the_callers_deadline(case, tmp_path, monkeypatch):
    built = await _build(case, "recorded", tmp_path, monkeypatch)

    await _exercise(built.backend, case, deadline=2.5)

    assert built.transport.sent
    assert all(sent.deadline <= 2.5 for sent in built.transport.sent)


@pytest.mark.parametrize("case", CASES, ids=_ids(CASES))
async def test_a_spent_deadline_answers_without_raising(case, tmp_path, monkeypatch):
    built = await _build(case, case.modes[0], tmp_path, monkeypatch)

    answers = await _exercise(built.backend, case, deadline=0.0)

    for method, answer in answers:
        assert isinstance(answer, _RETURNS[method]), (method, answer)
    if case.http:
        # A service backend has nothing to answer from but the service.
        for method, answer in answers:
            assert (answer.reason or "").startswith("no answer within"), (method, answer)
        assert all(page.outcome is Outcome.UNAVAILABLE for page in _pages(answers))


@pytest.mark.parametrize("case", ALARM_STATE_CASES, ids=_ids(ALARM_STATE_CASES))
async def test_a_read_before_catch_up_waits_and_answers_a_record(case, tmp_path, monkeypatch):
    built = await _build(case, case.modes[0], tmp_path, monkeypatch)
    starting = asyncio.create_task(built.backend.start())
    await asyncio.sleep(0)
    try:
        state = await built.backend.state_for(case.pv, DEADLINE)
    finally:
        await starting
        await built.backend.aclose()

    assert isinstance(state, AlarmStateRecord), state


# --- every request is a read or a search ------------------------------------------


@pytest.mark.parametrize("mode", ["recorded", "empty"])
@pytest.mark.parametrize("case", HTTP_CASES, ids=_ids(HTTP_CASES))
async def test_every_request_is_an_allowed_read_or_search(case, mode, tmp_path, monkeypatch):
    built = await _build(case, mode, tmp_path, monkeypatch)

    await _exercise(built.backend, case)

    assert built.transport.sent
    assert all(
        _is_read_or_search(sent.method, sent.path, sent.json_body) for sent in built.transport.sent
    )
    built.assert_read_only()


@pytest.mark.parametrize(
    ("method", "path", "body"),
    [
        ("PUT", "/resources/channels/X", {}),
        ("DELETE", "/resources/channels/X", None),
        ("POST", "/resources/channels", []),
        ("GET", "/search", {"x": 1}),
    ],
)
async def test_the_recording_transport_fails_a_request_that_writes(method, path, body):
    transport = RecordingTransport(_always([]))

    with pytest.raises(pytest.fail.Exception):
        await transport.request(method, path, json_body=body, deadline=DEADLINE)

    assert transport.violations == [f"{method} {path}"]
    assert transport.sent == []


async def test_the_kafka_feed_reads_through_a_consumer_and_never_produces(monkeypatch):
    made: dict[str, Any] = {}

    class AIOKafkaConsumer(ReadOnlyConsumer):
        def __init__(self, *topics: str, **options: Any) -> None:
            super().__init__(_FEED)
            made["consumer"] = (topics, options, self)

    class AIOKafkaProducer:
        def __init__(self, *_args: Any, **_options: Any) -> None:
            made["producer"] = True
            pytest.fail("the alarm feed built a producer")

    aiokafka = types.ModuleType("aiokafka")
    aiokafka.AIOKafkaConsumer = AIOKafkaConsumer  # type: ignore[attr-defined]
    aiokafka.AIOKafkaProducer = AIOKafkaProducer  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "aiokafka", aiokafka)
    spec = Spec(
        kind="alarm_state",
        type="phoebus_alarm_server",
        target="live",
        settings={"bootstrap_servers": "broker.demo.lab:9092", "topic": "Demo"},
        rehearsal=False,
    )
    backend = await ConnectorFactory.create_alarm_state_connector(spec)

    await backend.start()
    try:
        state = await backend.state_for(PV, DEADLINE)
    finally:
        await backend.aclose()

    assert isinstance(state, AlarmStateRecord)
    assert "producer" not in made
    topics, options, consumer = made["consumer"]
    assert topics == ("Demo",)
    assert options["group_id"] is None
    assert options["enable_auto_commit"] is False
    assert consumer.violations == []
    assert consumer.stopped


# --- login pages, over the wire ---------------------------------------------------

_LOGIN_PAGE = b"<!doctype html><html><body><form>Sign in</form></body></html>"


class _LoginProxy:
    """A loopback access proxy: every path answers its login page or a redirect to it."""

    def __init__(self) -> None:
        self.redirect = False
        self.received: list[tuple[str, str]] = []
        self.url = ""


@pytest.fixture
def login_proxy() -> Iterator[_LoginProxy]:
    proxy = _LoginProxy()

    class Handler(BaseHTTPRequestHandler):
        def _reply(self) -> None:
            self.rfile.read(int(self.headers.get("Content-Length") or 0))
            proxy.received.append((self.command, urlsplit(self.path).path))
            if proxy.redirect:
                self.send_response(302)
                self.send_header("Location", f"{proxy.url}/sso/login")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(_LOGIN_PAGE)))
            self.end_headers()
            self.wfile.write(_LOGIN_PAGE)

        do_GET = do_POST = do_PUT = do_DELETE = do_PATCH = _reply

        def log_message(self, *args: object) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    proxy.url = f"http://127.0.0.1:{server.server_address[1]}"
    thread = threading.Thread(target=server.serve_forever, args=(0.01,), daemon=True)
    thread.start()
    yield proxy
    server.shutdown()
    server.server_close()
    thread.join()


@pytest.mark.parametrize("redirect", [False, True], ids=["login-page", "redirect"])
@pytest.mark.parametrize("case", HTTP_CASES, ids=_ids(HTTP_CASES))
async def test_a_login_page_reads_as_the_access_proxy_on_every_method(case, redirect, login_proxy):
    login_proxy.redirect = redirect
    spec = Spec(
        kind=case.kind,  # type: ignore[arg-type]
        type=case.type,
        target="live",
        settings={"url": login_proxy.url},
        rehearsal=False,
    )
    create = _create(case.kind)
    backend = await create(spec)

    answers = await _exercise(backend, case)

    for method, answer in answers:
        if isinstance(answer, DirectoryPage | HistoryPage):
            assert answer.outcome is Outcome.UNAVAILABLE, method
            assert answer.reason == LOGIN_PAGE_REASON, method
        else:
            assert answer == Unavailable(LOGIN_PAGE_REASON), method
    assert login_proxy.received
    assert all(_is_read_or_search(method, path, None) for method, path in login_proxy.received)
    assert "/sso/login" not in {path for _, path in login_proxy.received}


# --- config to request ------------------------------------------------------------


async def test_the_configured_description_property_reaches_the_channelfinder_request():
    config = {
        "pv_directory": {
            "type": "channelfinder",
            "settings": {"url": URL},
            "description_property": "pvDescription",
        }
    }
    spec = service_targets(config, ariel_configured=False)["directory"]["live"]
    transport = RecordingTransport(_HTTP_ANSWERS["channelfinder"]["recorded"])

    backend = await ConnectorFactory.create_pv_directory_connector(spec, transport=transport)
    page = await backend.search(DirectoryQuery(words="cooling water"), DEADLINE)

    first = transport.sent[0]
    assert (first.method, first.path) == ("GET", CF_CHANNELS)
    assert ("pvDescription", "*cooling*water*") in first.params
    assert "pvDescription=*cooling*water*" in page.request
    assert not any(key == "recordDesc" for sent in transport.sent for key, _ in sent.params)
