"""The ``alarms`` route under ``/api/directory``: every channel in alarm now.

Every test runs the real lifespan and the real ``DirectoryService`` over a config;
the alarm feed is the demo replay through the real factory, with its load moment
fixed so the rebased times are the shared fixture's.
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
_replay = pytest.importorskip("osprey_connectors.alarm_state._replay")

from fastapi.testclient import TestClient  # noqa: E402

from osprey_connectors import control_context  # noqa: E402

_CONFIG_SEAM = "osprey_connectors.workspace.load_osprey_config"
_FIXTURE = Path(__file__).parent / "fixtures" / "directory_routes.json"
_LOAD_MOMENT = datetime(2026, 11, 6, 12, 0, 0, tzinfo=UTC)

VA_BUILD = {"type": "virtual_accelerator"}
LIVE_BUILD = {"type": "epics"}
FACILITY = {"type": "channelfinder", "settings": {"url": "http://cf.example"}}
DEMO_FEED = {"alarm_state": {"replay_file": "demo:example-alarms"}}


@pytest.fixture(autouse=True)
def _seams(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("OSPREY_CONFIG", str(tmp_path / "config.yml"))
    monkeypatch.setattr(_replay, "_now", lambda: _LOAD_MOMENT)


def _config(tmp_path: Path, control_system: dict[str, Any], section: dict[str, Any]) -> Any:
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
    return config


@pytest.fixture()
def start(mock_registry, monkeypatch: pytest.MonkeyPatch):  # noqa: ARG001
    """Create the app over a config, on *target*, and enter its lifespan."""

    @contextmanager
    def started(config: dict[str, Any], target: str) -> Iterator[TestClient]:
        record = control_context.ControlContext(target=target, generation=1)
        monkeypatch.setattr(control_context, "read_record", lambda **_: record)
        monkeypatch.setattr(_CONFIG_SEAM, lambda: config)
        from osprey.interfaces.channel_finder.app import create_app

        with TestClient(create_app(project_cwd="/tmp/test-project")) as client:
            yield client

    return started


def _alarms(client: TestClient, query: str = "") -> dict[str, Any]:
    response = client.get(f"/api/directory/alarms{query}")
    assert response.status_code == 200
    return response.json()


def test_the_route_answers_the_shared_fixture_on_the_demo_replay(start, tmp_path: Path) -> None:
    with start(_config(tmp_path, VA_BUILD, DEMO_FEED), "va") as client:
        answer = _alarms(client)

    assert answer == json.loads(_FIXTURE.read_text(encoding="utf-8"))["alarms"]


def test_a_severity_filter_and_a_limit_of_zero(start, tmp_path: Path) -> None:
    with start(_config(tmp_path, VA_BUILD, DEMO_FEED), "va") as client:
        minor = _alarms(client, "?severity=MINOR")
        badge = _alarms(client, "?limit=0")
        too_many = client.get("/api/directory/alarms?limit=1001")

    assert [row["severity"] for row in minor["rows"]] == ["MINOR"]
    assert badge["rows"] == []
    assert badge["counts"] == minor["counts"]
    assert too_many.status_code == 422


def test_a_target_with_no_alarm_state_source_answers_the_no_source_form(
    start, tmp_path: Path
) -> None:
    with start(_config(tmp_path, LIVE_BUILD, FACILITY), "live") as client:
        answer = _alarms(client)

    assert answer == {
        "outcome": "unavailable",
        "source": {"target": "live", "service": "none", "label": "All PVs", "rehearsal": False},
        "reason": "not available for this target",
        "caveats": [],
    }
