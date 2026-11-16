"""Tests for the gated ``active_alarms`` tool, in process on a fake alarm feed.

The tool reads the facts' list of every channel in alarm; the facts reach the
feed only through the directory service, whose factory create call each test
swaps for a fake, or for the real one over the demo replay.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("osprey_connectors.service_answers")

import osprey.channel_roster as channel_roster
from osprey.mcp_server.pv_directory import server
from osprey.mcp_server.pv_directory.tools import active_alarms
from osprey.services.pv_directory import _connectors
from osprey.services.pv_directory.facts import ACTIVE_ALARMS_MAX_ROWS, panel_link
from osprey_connectors import control_context
from osprey_connectors.alarm_state import (
    AlarmStateRecord,
    AlarmStateRow,
    AlarmStates,
)
from osprey_connectors.factory import ConnectorFactory
from osprey_connectors.service_answers import Source, Unavailable
from tests._service_factory import patch_create_service
from tests.mcp_server.conftest import assert_raises_error

_REAL_CREATE = ConnectorFactory.create_service

VA_BUILD = {"type": "virtual_accelerator"}
LIVE_BUILD = {"type": "epics"}
KAFKA = {
    "type": "phoebus_alarm_server",
    "settings": {"bootstrap_servers": "kafka:9092", "topic": "Accelerator"},
}
FEED = {"alarm_state": KAFKA}


class FakeAlarmState:
    """A caught-up feed; ``listed`` is what its list read answers."""

    listed: Any = None

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

    async def start(self) -> None:
        pass

    async def aclose(self) -> None:
        pass

    async def states(self, deadline: float) -> Any:  # noqa: ARG002 - a fake
        return self.listed


@pytest.fixture(autouse=True)
def fake_feed(monkeypatch: pytest.MonkeyPatch) -> None:
    async def create(spec: Any, *, project_dir: Path | None = None, **seams: Any) -> Any:
        return FakeAlarmState(spec, project_dir=project_dir, **seams)

    patch_create_service(monkeypatch, "alarm_state", create)
    monkeypatch.setattr(FakeAlarmState, "listed", None)


@pytest.fixture(autouse=True)
def target(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    holder: list[Any] = [None]
    monkeypatch.setattr(control_context, "read_record", lambda **_: holder[0])
    return holder


@pytest.fixture(autouse=True)
def cold_roster() -> Iterator[None]:
    channel_roster._roster_cache.clear()
    yield
    channel_roster._roster_cache.clear()
    server.set_tool_context(None)


def _install(tmp_path: Path, control_system: dict[str, Any], section: dict[str, Any]) -> Any:
    (tmp_path / "facility.json").write_text(json.dumps({"channels": []}), encoding="utf-8")
    config: dict[str, Any] = {
        "config_dir": str(tmp_path),
        "control_system": control_system,
        "pv_directory": section,
    }
    if control_system == VA_BUILD:
        config["services"] = {"virtual_accelerator": {"port": 5064}}
    context = server.build_tool_context(config, project_dir=tmp_path)
    server.set_tool_context(context)
    return context


def _row(channel: str, severity: str, time: str) -> Any:
    state = AlarmStateRecord(severity=severity, time=time)
    return AlarmStateRow(channel=channel, path=f"/SR/{channel}", state=state)


def _listed() -> Any:
    rows = (
        _row("SR:A", "MINOR", "2026-11-11T07:00:00.000Z"),
        _row("SR:B", "MAJOR", "2026-11-11T06:00:00.000Z"),
        _row("SR:C", "OK", "2026-11-11T07:30:00.000Z"),
    )
    return AlarmStates(rows=rows, stale=False, last_contact="2026-11-11T08:00:00.000Z")


async def test_rows_carry_their_panel_link_and_the_counts_cover_every_channel(
    tmp_path: Path,
) -> None:
    FakeAlarmState.listed = _listed()
    _install(tmp_path, LIVE_BUILD, FEED)

    result = json.loads(await active_alarms.active_alarms())

    assert result["outcome"] == "found"
    assert result["counts"] == {"MAJOR": 1, "MINOR": 1, "OK": 1}
    assert result["total"] == 3
    assert [row["pv"] for row in result["rows"]] == ["SR:B", "SR:A"]
    assert result["rows"][0]["links"] == panel_link("pv", "SR:B")
    assert result["rows"][0]["links"] == {"panel": "/panel/channel-finder/#/pv/SR%3AB"}


async def test_a_severity_filter_and_a_limit(tmp_path: Path) -> None:
    FakeAlarmState.listed = _listed()
    _install(tmp_path, LIVE_BUILD, FEED)

    minor = json.loads(await active_alarms.active_alarms(severities=["MINOR"]))
    first = json.loads(await active_alarms.active_alarms(limit=1))

    assert [row["pv"] for row in minor["rows"]] == ["SR:A"]
    assert [row["pv"] for row in first["rows"]] == ["SR:B"]
    assert first["truncated"] is True
    assert first["counts"] == minor["counts"]


async def test_a_limit_out_of_range_is_refused(tmp_path: Path) -> None:
    _install(tmp_path, LIVE_BUILD, FEED)

    with assert_raises_error(error_type="validation_error"):
        await active_alarms.active_alarms(limit=-1)
    with assert_raises_error(error_type="validation_error"):
        await active_alarms.active_alarms(limit=ACTIVE_ALARMS_MAX_ROWS + 1)


async def test_a_feed_that_cannot_answer_refuses_service_unavailable(tmp_path: Path) -> None:
    FakeAlarmState.listed = Unavailable("alarm feed still loading")
    _install(tmp_path, LIVE_BUILD, FEED)

    with assert_raises_error(error_type="service_unavailable") as refused:
        await active_alarms.active_alarms()

    assert refused["envelope"]["details"]["reason"] == "alarm feed still loading"


async def test_no_source_on_the_target_refuses_not_supported(
    tmp_path: Path, target: list[Any]
) -> None:
    target[0] = control_context.ControlContext(target="va", generation=1)
    _install(
        tmp_path, VA_BUILD, {"type": "channelfinder", "settings": {"url": "http://cf"}, **FEED}
    )

    with assert_raises_error(error_type="not_supported") as refused:
        await active_alarms.active_alarms()

    expected = _connectors.unavailable_payload(_connectors.NO_SOURCE_REASON, target="va")
    assert {key: refused["envelope"]["details"][key] for key in expected} == expected


async def test_the_demo_replay_lists_its_two_active_channels(
    tmp_path: Path, target: list[Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("osprey_connectors.alarm_state._replay")
    patch_create_service(monkeypatch, "alarm_state", _REAL_CREATE)
    target[0] = control_context.ControlContext(target="va", generation=1)
    context = _install(tmp_path, VA_BUILD, {"alarm_state": {"replay_file": "demo:example-alarms"}})

    try:
        result = json.loads(await active_alarms.active_alarms())
    finally:
        await context.service.aclose()

    assert result["source"]["target"] == "va"
    assert [row["severity"] for row in result["rows"]] == ["MAJOR", "MINOR"]


def test_the_docstring_says_what_stale_means() -> None:
    doc = active_alarms.active_alarms.__doc__ or ""
    assert "stale" in doc
    assert "every channel" in doc
