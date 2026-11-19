"""Tests for the gated ``server_fleet`` tool, in process over a mocked fleet service.

The tool reads one thing: the fleet service's snapshot of the control target,
through the fleet connection. Each test answers that request with an
``httpx.MockTransport``; the directory service in the tool context refuses every
use, so a test that passes proves the tool queried no directory and read no
channel.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from typing import Any

import httpx
import pytest

pytest.importorskip("osprey_connectors.safety_file")

from fastmcp import FastMCP

from osprey import server_fleet_connection
from osprey.mcp_server.channel_directory import server
from osprey.mcp_server.channel_directory.tools import server_fleet
from osprey.server_fleet_connection import SERVER_FLEET_URL_ENV_VAR
from osprey.services.channel_directory.facts import panel_link
from osprey.services.channel_directory.fleet.snapshot import (
    Liveness,
    Outcome,
    ReadPathStatus,
    ServerFleetRow,
    Snapshot,
    SourceStatus,
)
from osprey.services.channel_directory.fleet.states import (
    Finding,
    LivenessKind,
    ReadPath,
    State,
)
from osprey_connectors import control_context
from tests.mcp_server.conftest import assert_raises_error

NOW = 1_700_000_000.0
LIVE = {"type": "epics"}
FLEET_CONFIG = {"control_system": LIVE, "channel_directory": {"fleet": {}}}


class NoDirectory:
    """A directory service and facts that fail the test on any use."""

    def __getattr__(self, name: str) -> Any:
        raise AssertionError(f"server_fleet touched the directory ({name})")


def _row(
    name: str,
    state: State = State.ALIVE,
    *,
    host: str = "host-1",
    class_name: str = "default",
    group: str | None = None,
    findings: tuple[Finding, ...] = (),
) -> ServerFleetRow:
    return ServerFleetRow(
        name=name,
        host=host,
        class_name=class_name,
        group=group,
        state=state,
        since=NOW - 100,
        reason=f"{name} is {state.value}",
        findings=findings,
        directory_status="Active",
        listed=True,
        sources=(),
        liveness=Liveness(channel=f"{name}:HB", kind=LivenessKind.COUNTER, value=3, read_at=NOW),
    )


ROWS = [
    _row("ioc-a", host="host-1", group="rf"),
    _row("ioc-b", State.SILENT, host="host-2", group="rf"),
    _row("ioc-c", State.UNREACHED, host="host-2", class_name="vacuum"),
    _row("ioc-d", host="host-3", findings=(Finding.DEGRADED,)),
    _row("ioc-e", State.STALLED, host="host-1"),
]


def _snapshot(target: str = "live", **overrides: Any) -> Snapshot:
    fields: dict[str, Any] = {
        "target": target,
        "generated_at": NOW - 10,
        "cycle": 4,
        "period_s": 30.0,
        "cycle_duration_s": 2.0,
        "read_path": ReadPathStatus(status=ReadPath.OK, channel="SR:PROBE"),
        "sources": (SourceStatus(name="directory", outcome="ok"),),
        "group_by": "host",
        "servers": ROWS,
    }
    fields.update(overrides)
    return Snapshot.ok(**fields)


@pytest.fixture(autouse=True)
def _no_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(SERVER_FLEET_URL_ENV_VAR, raising=False)


@pytest.fixture(autouse=True)
def control_record(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    holder: list[Any] = [None]
    monkeypatch.setattr(control_context, "read_record", lambda **_: holder[0])
    return holder


@pytest.fixture(autouse=True)
def context() -> Iterator[None]:
    server.set_tool_context(
        server.ToolContext(
            predicates={"channel_directory_fleet": True},
            service=NoDirectory(),  # type: ignore[arg-type]
            channel_facts=NoDirectory(),  # type: ignore[arg-type]
            config=FLEET_CONFIG,
        )
    )
    yield
    server.set_tool_context(None)


@pytest.fixture()
def serve(monkeypatch: pytest.MonkeyPatch) -> Callable[..., list[httpx.Request]]:
    """Answer the tool's snapshot request with *handler*; returns the requests seen."""

    def install(handler: Callable[[httpx.Request], httpx.Response]) -> list[httpx.Request]:
        seen: list[httpx.Request] = []

        def record(request: httpx.Request) -> httpx.Response:
            seen.append(request)
            return handler(request)

        real = server_fleet_connection.fetch_snapshot
        transport = httpx.MockTransport(record)

        async def fetch(config: Any, target: str) -> Any:
            return await real(config, target, transport=transport, now=NOW)

        monkeypatch.setattr(server_fleet_connection, "fetch_snapshot", fetch)
        return seen

    return install


def _document(document: dict[str, Any]) -> Callable[[httpx.Request], httpx.Response]:
    return lambda request: httpx.Response(200, json=document)


async def _call(**arguments: Any) -> dict[str, Any]:
    return json.loads(await server_fleet.server_fleet(**arguments))


# --- an ok snapshot ---------------------------------------------------------------


async def test_unfiltered_answer_carries_counts_read_path_sources_and_time(serve) -> None:
    seen = serve(_document(_snapshot().to_json()))

    result = await _call()

    assert result["outcome"] == "ok"
    assert result["target"] == "live"
    assert result["statement"] is None and result["reason"] is None
    assert result["generated_at"] == NOW - 10
    assert result["snapshot_time"] == "2023-11-14T22:13:10+00:00"
    assert result["read_path"] == {"status": "ok", "channel": "SR:PROBE"}
    assert result["sources"] == [{"name": "directory", "outcome": "ok", "reason": None}]
    assert result["counts"]["by_state"] == {
        "alive": 2,
        "stalled": 1,
        "silent": 1,
        "unreached": 1,
        "unknown": 0,
    }
    assert result["counts"]["by_finding"]["degraded"] == 1
    assert result["total"] == result["matched"] == 5
    assert result["truncated"] is False
    assert [request.url.params["target"] for request in seen] == ["live"]


async def test_rows_are_the_snapshots_rows_in_its_order_each_with_its_server_page(serve) -> None:
    snapshot = _snapshot()
    serve(_document(snapshot.to_json()))

    result = await _call()

    expected = snapshot.to_json()["servers"]
    assert [{k: v for k, v in row.items() if k != "links"} for row in result["rows"]] == expected
    for row in result["rows"]:
        assert row["links"] == panel_link("server", row["name"])
    first = expected[0]["name"]
    assert result["rows"][0]["links"] == {"panel": f"/panel/channel-finder/#/server/{first}"}


async def test_every_answer_states_each_states_meaning_and_links_the_view(serve) -> None:
    serve(_document(_snapshot().to_json()))

    result = await _call()

    assert list(result["meanings"]) == [state.value for state in State]
    for meaning in result["meanings"].values():
        assert meaning and "\n" not in meaning
    assert result["meanings"]["silent"].startswith("it gave no answer now")
    assert result["links"] == server_fleet.VIEW_LINK
    assert result["links"]["panel"].startswith("/panel/channel-finder/#/")


@pytest.mark.parametrize(
    ("arguments", "names"),
    [
        ({"state": ["silent", "unreached"]}, {"ioc-b", "ioc-c"}),
        ({"finding": ["degraded"]}, {"ioc-d"}),
        ({"host": "host-2"}, {"ioc-b", "ioc-c"}),
        ({"group": "host-1"}, {"ioc-a", "ioc-e"}),
        ({"text": "VACUUM"}, {"ioc-c"}),
        ({"text": "ioc-e"}, {"ioc-e"}),
        ({"state": ["alive"], "host": "host-3"}, {"ioc-d"}),
        ({"state": ["unknown"]}, set()),
    ],
)
async def test_filters_select_rows(serve, arguments: dict[str, Any], names: set[str]) -> None:
    serve(_document(_snapshot().to_json()))

    result = await _call(**arguments)

    assert {row["name"] for row in result["rows"]} == names
    assert result["matched"] == len(names)
    assert "counts" not in result and "sources" not in result and "total" not in result
    assert result["filters"]
    assert result["generated_at"] == NOW - 10


async def test_group_filters_on_the_snapshots_grouping_field(serve) -> None:
    serve(_document(_snapshot(group_by="group").to_json()))

    result = await _call(group="rf")

    assert {row["name"] for row in result["rows"]} == {"ioc-a", "ioc-b"}


async def test_limit_cuts_rows_and_sets_truncated(serve) -> None:
    serve(_document(_snapshot().to_json()))

    result = await _call(limit=2)

    assert len(result["rows"]) == 2
    assert result["matched"] == 5
    assert result["truncated"] is True
    assert result["counts"]["by_state"]["alive"] == 2


async def test_the_control_target_is_asked_for(serve, control_record) -> None:
    control_record[0] = control_context.ControlContext(target="va", generation=3)
    seen = serve(_document(_snapshot(target="va").to_json()))

    result = await _call()

    assert result["target"] == "va"
    assert [request.url.params["target"] for request in seen] == ["va"]


# --- outcomes that are not ok -----------------------------------------------------


def _assert_no_server_called_down(result: dict[str, Any]) -> None:
    assert result["statement"]
    assert " down" not in result["statement"].lower()
    assert "rows" not in result
    assert result["meanings"] and result["links"] == server_fleet.VIEW_LINK


@pytest.mark.parametrize(
    "outcome", [Outcome.NO_SNAPSHOT, Outcome.NOT_REFRESHED, Outcome.CANNOT_LIST]
)
async def test_a_service_outcome_is_a_plain_statement(serve, outcome: Outcome) -> None:
    reason = f"because {outcome.value}"
    serve(_document(Snapshot.not_ok(outcome, target="live", reason=reason).to_json()))

    result = await _call()

    assert result["outcome"] == outcome.value
    assert result["reason"] == reason
    assert "live" in result["statement"]
    _assert_no_server_called_down(result)


@pytest.mark.parametrize(
    ("control_system", "servers", "server_word"),
    [
        (LIVE, "IOCs", "IOC"),
        ({"type": "tango"}, "device servers", "device server"),
        ({"type": "facility.Custom"}, "servers", "server"),
    ],
)
async def test_a_statement_names_servers_in_the_targets_words(
    serve, control_system: dict[str, Any], servers: str, server_word: str
) -> None:
    server.set_tool_context(
        server.ToolContext(
            predicates={"channel_directory_fleet": True},
            service=NoDirectory(),  # type: ignore[arg-type]
            channel_facts=NoDirectory(),  # type: ignore[arg-type]
            config={**FLEET_CONFIG, "control_system": control_system},
        )
    )
    serve(_document(Snapshot.not_ok(Outcome.CANNOT_LIST, target="live", reason="r").to_json()))

    statement = (await _call())["statement"]

    assert f"cannot list the {servers} of target live" in statement
    assert f"whether any {server_word} runs" in statement
    if control_system != LIVE:
        assert "IOC" not in statement and "PV" not in statement


async def test_an_unreachable_service_is_a_plain_statement_not_an_error(serve) -> None:
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    serve(refuse)

    result = await _call(state=["silent"])

    assert result["outcome"] == "unavailable"
    assert "could not reach" in result["reason"]
    _assert_no_server_called_down(result)


async def test_a_wrong_answer_is_unavailable(serve) -> None:
    serve(lambda request: httpx.Response(503))

    result = await _call()

    assert result["outcome"] == "unavailable"
    _assert_no_server_called_down(result)


async def test_a_stale_snapshot_keeps_its_rows_and_says_they_are_old(serve) -> None:
    serve(_document(_snapshot(generated_at=NOW - 1000).to_json()))

    result = await _call()

    assert result["outcome"] == "stale"
    assert "stale" in result["statement"]
    assert result["snapshot_time"] in result["statement"]
    assert " down" not in result["statement"].lower()
    assert result["generated_at"] == NOW - 1000
    assert len(result["rows"]) == 5
    assert result["counts"]["by_state"]["silent"] == 1


async def test_with_the_fleet_off_the_answer_is_a_statement() -> None:
    server.set_tool_context(
        server.ToolContext(
            predicates={},
            service=NoDirectory(),  # type: ignore[arg-type]
            channel_facts=NoDirectory(),  # type: ignore[arg-type]
            config={"control_system": LIVE},
        )
    )

    result = await _call()

    assert result["outcome"] == "unavailable"
    _assert_no_server_called_down(result)


# --- argument errors --------------------------------------------------------------


@pytest.mark.parametrize("limit", [-1, server_fleet.SERVER_FLEET_MAX_ROWS + 1])
async def test_a_limit_out_of_range_is_refused(limit: int) -> None:
    with assert_raises_error(error_type="validation_error"):
        await server_fleet.server_fleet(limit=limit)


@pytest.mark.parametrize("arguments", [{"state": ["down"]}, {"finding": ["broken"]}])
async def test_an_unknown_state_or_finding_is_refused(arguments: dict[str, Any]) -> None:
    with assert_raises_error(error_type="validation_error") as ctx:
        await server_fleet.server_fleet(**arguments)
    assert "unknown" in ctx["envelope"]["error_message"]


# --- registration -----------------------------------------------------------------


async def test_register_adds_one_read_only_tool() -> None:
    probe = FastMCP("server-fleet-probe")
    server_fleet.register(probe)

    (tool,) = await probe.list_tools()

    assert tool.name == "server_fleet"
    assert tool.annotations.readOnlyHint is True


def test_the_tool_registers_only_when_the_fleet_predicate_holds() -> None:
    def registered(predicate: bool) -> list[str]:
        probe = FastMCP("server-fleet-gate")
        context = server.ToolContext(
            predicates=dict.fromkeys(
                (
                    "channel_directory_alarm_state",
                    "channel_directory_alarm_log",
                    "channel_directory_alarm_config",
                    "channel_directory_put_log",
                ),
                False,
            )
            | {"channel_directory_fleet": predicate},
            service=NoDirectory(),  # type: ignore[arg-type]
            channel_facts=NoDirectory(),  # type: ignore[arg-type]
        )
        return server.register_tools(probe, context)

    assert registered(True) == ["server_fleet"]
    assert registered(False) == []
