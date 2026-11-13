"""Tests for the PV directory MCP server and its search tools, in process on fake backends.

The directory service builds every backend through the factory's create API, so
each test swaps that call for one returning a fake directory. The control target
is the deployment's baseline: the control-context record is stubbed to none.
"""

from __future__ import annotations

import asyncio
import dataclasses
import importlib
import inspect
import json
import sys
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType
from typing import Any
from urllib.parse import unquote

import pytest

pytest.importorskip("osprey_connectors.service_answers")

from fastmcp import FastMCP

import osprey.channel_roster as channel_roster
from osprey.mcp_server.pv_directory import server
from osprey.mcp_server.pv_directory.tools import properties, search
from osprey.registry.mcp import FRAMEWORK_SERVERS
from osprey.services.pv_directory import _connectors, facts, service
from osprey_connectors import control_context
from osprey_connectors.alarm_state import AlarmStateRecord, AlarmStateRow, AlarmStates
from osprey_connectors.directory import DirectoryPage, DirectoryRow
from osprey_connectors.service_answers import Outcome, Source, Unavailable
from tests._service_factory import patch_create_service
from tests.mcp_server.conftest import assert_raises_error

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

LIVE_BUILD = {"type": "epics"}
FACILITY = {"type": "channelfinder", "settings": {"url": "http://cf.example", "timeout_s": 0.3}}


class FakeDirectory:
    """A directory over fixed rows; ``words`` is a substring of the name."""

    rows: list[Any] = []
    values: dict[str, list[str]] = {}
    count_delay = 0.0
    failure: str | None = None

    def __init__(self, spec: Any, *, project_dir: Path | None = None, **seams: Any) -> None:  # noqa: ARG002
        self.spec = spec
        self.asked: list[tuple[str, str | None]] = []

    @property
    def source(self) -> Any:
        return Source(
            kind=self.spec.kind,
            target=self.spec.target,
            service=self.spec.type,
            rehearsal=self.spec.rehearsal,
        )

    def _hits(self, query: Any) -> list[Any]:
        return [row for row in self.rows if query.words in row.name]

    async def search(self, query: Any, deadline: float) -> Any:  # noqa: ARG002 - a fake
        if self.failure:
            raise RuntimeError(self.failure)
        hits = self._hits(query)
        start = query.page * query.page_size
        rows = hits[start : start + query.page_size]
        return DirectoryPage(
            outcome=Outcome.FOUND if hits else Outcome.NO_MATCH,
            searched="from the backend",
            request=f"GET /fake?q={query.words}",
            source=self.source,
            total=len(hits),
            rows=rows,
            truncated=start + len(rows) < len(hits),
        )

    async def count(self, query: Any, deadline: float) -> Any:  # noqa: ARG002 - a fake
        if self.failure:
            raise RuntimeError(self.failure)
        await asyncio.sleep(self.count_delay)
        return len(self._hits(query))

    async def describe(self, name: str, deadline: float) -> Any:  # noqa: ARG002 - a fake
        return Outcome.NOT_LISTED

    async def property_values(self, name: str, prefix: str | None, deadline: float) -> Any:  # noqa: ARG002
        self.asked.append((name, prefix))
        if name not in self.values:
            return Unavailable(f"property {name!r} did not answer")
        return [value for value in self.values[name] if value.startswith(prefix or "")]


@pytest.fixture(autouse=True)
def built(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    """Every directory the service builds; the create call returns a fake."""
    made: list[Any] = []

    async def create(spec: Any, *, project_dir: Path | None = None, **seams: Any) -> Any:
        backend = FakeDirectory(spec, project_dir=project_dir, **seams)
        made.append(backend)
        return backend

    patch_create_service(monkeypatch, "directory", create)
    monkeypatch.setattr(FakeDirectory, "rows", [])
    monkeypatch.setattr(FakeDirectory, "values", {})
    monkeypatch.setattr(FakeDirectory, "count_delay", 0.0)
    monkeypatch.setattr(FakeDirectory, "failure", None)
    monkeypatch.setattr(control_context, "read_record", lambda **_: None)
    return made


@pytest.fixture(autouse=True)
def cold_roster() -> Iterator[None]:
    channel_roster._roster_cache.clear()
    yield
    channel_roster._roster_cache.clear()


@pytest.fixture(autouse=True)
def no_context() -> Iterator[None]:
    yield
    server.set_tool_context(None)


def _config(tmp_path: Path, section: dict[str, Any] | None, *curated: str) -> dict[str, Any]:
    channels = [{"id": address, "role": "none"} for address in curated]
    (tmp_path / "facility.json").write_text(json.dumps({"channels": channels}), encoding="utf-8")
    config: dict[str, Any] = {"config_dir": str(tmp_path), "control_system": LIVE_BUILD}
    if section is not None:
        config["pv_directory"] = section
    return config


def _install(tmp_path: Path, section: dict[str, Any] | None, *curated: str) -> server.ToolContext:
    context = server.build_tool_context(_config(tmp_path, section, *curated), project_dir=tmp_path)
    server.set_tool_context(context)
    return context


def _row(name: str, **fields: Any) -> Any:
    return DirectoryRow(name=name, **fields)


# -- directory_search ---------------------------------------------------------


async def test_search_rows_carry_their_origin_and_panel_link(tmp_path: Path) -> None:
    FakeDirectory.rows = [_row("SR:BPM1", server="ioc-a"), _row("SR:BPM2", server="ioc-b")]
    _install(tmp_path, FACILITY, "SR:BPM1")

    result = json.loads(await search.directory_search(words="SR"))

    assert result["outcome"] == "found"
    assert [(row["name"], row["origin"], row["links"]) for row in result["rows"]] == [
        ("SR:BPM1", "curated", {"panel": "/panel/channel-finder/#/pv/SR%3ABPM1"}),
        ("SR:BPM2", "directory", {"panel": "/panel/channel-finder/#/pv/SR%3ABPM2"}),
    ]
    assert result["source"]["service"] == "channelfinder"
    assert result["searched"] == "name or description matches SR"
    assert "note" not in result


async def test_a_link_round_trips_a_name_with_reserved_characters(tmp_path: Path) -> None:
    name = "SR:C01-MG{PS:QF}#1 x/y?z"
    FakeDirectory.rows = [_row(name)]
    _install(tmp_path, FACILITY)

    result = json.loads(await search.directory_search(words="SR"))

    link = result["rows"][0]["links"]["panel"]
    prefix = "/panel/channel-finder/#/pv/"
    assert link.startswith(prefix)
    segment = link.removeprefix(prefix)
    assert not set(segment) & set("/#? {}:")
    assert unquote(segment) == name


async def test_no_match_says_a_miss_is_not_an_absence(tmp_path: Path) -> None:
    FakeDirectory.rows = [_row("SR:BPM1")]
    _install(tmp_path, FACILITY)

    result = json.loads(await search.directory_search(words="QUAD"))

    assert result["outcome"] == "no_match"
    assert result["rows"] == []
    assert result["note"] == search.NO_MATCH_NOTE
    assert "does not show the channel does not exist" in result["note"]


async def test_a_late_count_stays_in_the_result_as_partial(tmp_path: Path) -> None:
    FakeDirectory.rows = [_row("SR:BPM1")]
    FakeDirectory.count_delay = 1.0
    _install(tmp_path, FACILITY)

    result = json.loads(await search.directory_search(words="SR"))

    assert result["outcome"] == "found"
    assert [row["name"] for row in result["rows"]] == ["SR:BPM1"]
    assert {entry["part"] for entry in result["partial"]} == {"count", "facets"}


class FakeFeed:
    """An alarm feed answering ``held``: a state per channel, or ``Unavailable``."""

    held: Any = {}

    def __init__(self, spec: Any, *, project_dir: Path | None = None, **seams: Any) -> None:  # noqa: ARG002
        self.spec = spec
        self.source = Source(
            kind=spec.kind, target=spec.target, service=spec.type, rehearsal=spec.rehearsal
        )

    async def start(self) -> None:
        pass

    async def aclose(self) -> None:
        pass

    async def states(self, deadline: float) -> Any:  # noqa: ARG002 - a fake
        if isinstance(self.held, Unavailable):
            return self.held
        rows = tuple(
            AlarmStateRow(channel=channel, path=None, state=state)
            for channel, state in self.held.items()
        )
        return AlarmStates(rows=rows, stale=False, last_contact=None)


_FEED = {
    "type": "phoebus_alarm_server",
    "settings": {"bootstrap_servers": "k:9092", "topic": "Accelerator"},
}


@pytest.fixture
def feed(monkeypatch: pytest.MonkeyPatch) -> type[FakeFeed]:
    async def create(spec: Any, *, project_dir: Path | None = None, **seams: Any) -> Any:
        return FakeFeed(spec, project_dir=project_dir, **seams)

    patch_create_service(monkeypatch, "alarm_state", create)
    monkeypatch.setattr(FakeFeed, "held", {})
    return FakeFeed


async def test_a_row_carries_the_alarm_feed_state(tmp_path: Path, feed: type[FakeFeed]) -> None:
    FakeDirectory.rows = [_row("SR:BPM1"), _row("SR:BPM2")]
    feed.held = {"SR:BPM1": AlarmStateRecord(severity="MINOR", status="HIGH_ALARM")}
    _install(tmp_path, {**FACILITY, "alarm_state": _FEED})

    result = json.loads(await search.directory_search(words="SR"))

    assert result["rows"][0]["alarm"]["severity"] == "MINOR"
    assert result["rows"][1]["alarm"] is None
    assert result["facets"]["alarm"] == [
        {"value": "MINOR", "count": 1},
        {"value": "none", "count": 1},
    ]


async def test_a_feed_without_an_answer_stays_in_the_result_as_partial(
    tmp_path: Path, feed: type[FakeFeed]
) -> None:
    FakeDirectory.rows = [_row("SR:BPM1")]
    feed.held = Unavailable("alarm feed still loading")
    _install(tmp_path, {**FACILITY, "alarm_state": _FEED})

    result = json.loads(await search.directory_search(words="SR"))

    assert "alarm" not in result["rows"][0]
    assert result["partial"] == [{"part": "alarm", "reason": "alarm feed still loading"}]


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


async def test_no_source_on_the_target_refuses_not_supported(tmp_path: Path) -> None:
    _install(tmp_path, {"display": {}})

    with assert_raises_error(error_type="not_supported") as refused:
        await search.directory_search(words="SR")

    envelope = refused["envelope"]
    expected = _connectors.unavailable_payload(_connectors.NO_SOURCE_REASON, target="live")
    assert envelope["details"] == _no_source_page(expected)
    assert envelope["error_message"] == f"directory_search: {_connectors.NO_SOURCE_REASON}"


async def test_a_directory_that_does_not_answer_refuses_service_unavailable(
    tmp_path: Path,
) -> None:
    FakeDirectory.failure = "connection refused"
    _install(tmp_path, FACILITY)

    with assert_raises_error(error_type="service_unavailable") as refused:
        await search.directory_search(words="SR")

    details = refused["envelope"]["details"]
    assert details["outcome"] == "unavailable"
    assert details["source"]["service"] == "channelfinder"
    assert details["reason"] == "the service failed: RuntimeError"
    assert "connection refused" not in json.dumps(details)


# -- directory_properties -----------------------------------------------------


async def test_properties_lists_the_values_with_their_source(
    tmp_path: Path, built: list[Any]
) -> None:
    FakeDirectory.values = {"iocName": ["ioc-a", "ioc-b", "sim-1"]}
    _install(tmp_path, FACILITY)

    result = json.loads(await properties.directory_properties("iocName", prefix="ioc-"))

    assert result["values"] == ["ioc-a", "ioc-b"]
    assert result["count"] == 2
    assert result["property"] == "iocName"
    assert result["prefix"] == "ioc-"
    assert result["source"]["service"] == "channelfinder"
    assert built[0].asked == [("iocName", "ioc-")]


async def test_properties_unavailable_from_the_service_refuses(tmp_path: Path) -> None:
    _install(tmp_path, FACILITY)

    with assert_raises_error(error_type="service_unavailable") as refused:
        await properties.directory_properties("nope")

    details = refused["envelope"]["details"]
    assert details["reason"] == "property 'nope' did not answer"
    assert details["source"]["service"] == "channelfinder"


async def test_properties_with_no_source_refuses_not_supported(tmp_path: Path) -> None:
    _install(tmp_path, {"display": {}})

    with assert_raises_error(error_type="not_supported") as refused:
        await properties.directory_properties("iocName")

    assert refused["envelope"]["details"]["reason"] == _connectors.NO_SOURCE_REASON


# -- the server ---------------------------------------------------------------


def test_the_server_and_its_tools_describe_themselves() -> None:
    assert server.mcp.name == "pv-directory"
    assert server.mcp.instructions.splitlines()[0].startswith("Read-only search")
    tools = {tool.name: tool for tool in asyncio.run(server.mcp.list_tools())}
    for name in ("directory_search", "directory_properties"):
        assert tools[name].title
        assert tools[name].annotations.readOnlyHint is True


def test_create_server_builds_the_context_from_the_rendered_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path, {**FACILITY, "alarm_log": {"type": "x", "settings": {"url": "u"}}})
    monkeypatch.delenv("OSPREY_CONFIG", raising=False)
    monkeypatch.setattr(server, "_rendered_config", lambda: (config, tmp_path))

    assert server.create_server() is server.mcp

    context = server.get_tool_context()
    assert isinstance(context.service, server.DirectoryService)
    assert isinstance(context.pv_facts, server.Facts)
    assert context.predicates["pv_directory_configured"] is True
    assert context.predicates["pv_directory_alarm_log"] is True
    assert context.predicates["pv_directory_put_log"] is False


def test_a_malformed_section_holds_no_predicate(tmp_path: Path) -> None:
    context = server.build_tool_context(
        _config(tmp_path, {"type": "channelfinder"}), project_dir=tmp_path
    )

    assert not any(context.predicates.values())


_OPTIONAL_TOOLS = FRAMEWORK_SERVERS["pv-directory"].optional_tools

_UNOFFERED = '''
def register(server):
    @server.tool()
    def unoffered_probe() -> str:
        """A module no optional_tools entry offers."""
        return "unoffered_probe"
'''


def _tool_names(mcp: FastMCP) -> set[str]:
    return {tool.name for tool in asyncio.run(mcp.list_tools())}


@pytest.mark.parametrize("tool", sorted(_OPTIONAL_TOOLS))
def test_a_gated_module_registers_only_when_its_optional_tools_key_holds(
    tmp_path: Path, tool: str
) -> None:
    context = server.build_tool_context(_config(tmp_path, FACILITY), project_dir=tmp_path)
    key = _OPTIONAL_TOOLS[tool].key
    held = dataclasses.replace(
        context, predicates={**dict.fromkeys(context.predicates, False), key: True}
    )
    withheld = dataclasses.replace(held, predicates={**held.predicates, key: False})

    on, off = FastMCP("probe-on"), FastMCP("probe-off")
    sharing = sorted(name for name, offer in _OPTIONAL_TOOLS.items() if offer.key == key)
    assert server.register_tools(on, held) == sharing
    assert server.register_tools(off, withheld) == []
    assert tool in _tool_names(on)
    assert not set(_OPTIONAL_TOOLS) & _tool_names(off)


def test_a_module_no_optional_tools_entry_offers_stays_unregistered(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from osprey.mcp_server.pv_directory import tools

    probes = tmp_path / "probes"
    probes.mkdir()
    (probes / "zz_unoffered.py").write_text(_UNOFFERED, encoding="utf-8")
    monkeypatch.setattr(tools, "__path__", [*tools.__path__, str(probes)])
    monkeypatch.delitem(sys.modules, f"{tools.__name__}.zz_unoffered", raising=False)
    context = server.build_tool_context(_config(tmp_path, FACILITY), project_dir=tmp_path)
    every = dataclasses.replace(context, predicates=dict.fromkeys(context.predicates, True))
    throwaway = FastMCP("probe")

    try:
        registered = server.register_tools(throwaway, every)
    finally:
        sys.modules.pop(f"{tools.__name__}.zz_unoffered", None)

    assert sorted(registered) == sorted(_OPTIONAL_TOOLS)
    assert "unoffered_probe" not in _tool_names(throwaway)


@pytest.mark.parametrize("tool", sorted(_OPTIONAL_TOOLS))
def test_a_gated_module_registers_its_tool_from_the_server_alone(tool: str) -> None:
    from osprey.mcp_server.pv_directory import tools

    module = importlib.import_module(f"{tools.__name__}.{tool}")
    assert not hasattr(module, "PREDICATE")
    assert list(inspect.signature(module.register).parameters) == ["server"]
    probe = FastMCP(f"{tool}-probe")

    module.register(probe)

    assert _tool_names(probe) == {tool}


# -- connectors without the directory modules ---------------------------------


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


async def test_modules_blocked_both_tools_refuse_with_the_upgrade_reason(
    tmp_path: Path, floor: ModuleType
) -> None:
    config = _config(tmp_path, FACILITY)
    directory = floor.DirectoryService(config, project_dir=tmp_path)
    server.set_tool_context(
        server.ToolContext(
            predicates={}, service=directory, pv_facts=facts.Facts(directory, config)
        )
    )

    with assert_raises_error(error_type="not_supported") as searched:
        await search.directory_search(words="SR")
    with assert_raises_error(error_type="not_supported") as listed:
        await properties.directory_properties("iocName")

    expected = _connectors.unavailable_payload(_connectors.FLOOR_REASON, target="live")
    assert searched["envelope"]["details"] == _no_source_page(expected)
    assert listed["envelope"]["details"]["reason"] == _connectors.FLOOR_REASON
