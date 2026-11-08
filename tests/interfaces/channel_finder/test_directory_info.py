"""The Channels panel's PV directory state: the ``/api/info`` descriptor and the mount.

Every test runs the real lifespan against a config, with the real
``DirectoryService`` behind it. The section's absence is today's panel exactly;
a malformed section is today's panel with a warning.
"""

from __future__ import annotations

import importlib
import json
import logging
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

contract = pytest.importorskip("osprey_connectors.pv_directory.contract")

from fastapi import APIRouter, FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from osprey.interfaces.channel_finder import directory_api  # noqa: E402
from osprey.interfaces.channel_finder.directory_values_ipc import ValueRow  # noqa: E402
from osprey.services.pv_directory import _connectors, service  # noqa: E402
from osprey_connectors import control_context  # noqa: E402
from osprey_connectors.factory import ConnectorFactory  # noqa: E402

_APP_LOGGER = "osprey.interfaces.channel_finder.app"
_CONFIG_SEAM = "osprey_connectors.workspace.load_osprey_config"
_FIXTURE = Path(__file__).parent / "fixtures" / "directory_routes.json"
_NEW_MODULES = (
    "osprey_connectors.pv_directory",
    "osprey_connectors.pv_directory.contract",
    "osprey_connectors.pv_directory.targets",
)

VA_BUILD = {"type": "virtual_accelerator"}
LIVE_BUILD = {"type": "epics"}
LABELLED = {"type": "labelled", "settings": {"url": "http://directory.example"}}


class LabelledDirectory(contract.PVDirectory):
    """A facility directory whose backend names itself."""

    label = "Facility PVs"

    async def search(self, query: Any, deadline: float) -> Any:
        raise NotImplementedError

    async def count(self, query: Any, deadline: float) -> Any:
        raise NotImplementedError

    async def describe(self, name: str, deadline: float) -> Any:
        raise NotImplementedError

    async def property_names(self, deadline: float) -> Any:
        raise NotImplementedError

    async def property_values(self, name: str, prefix: str | None, deadline: float) -> Any:
        raise NotImplementedError

    async def probe(self, deadline: float) -> Any:
        raise NotImplementedError


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
def labelled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(ConnectorFactory._pv_directory_connectors, "labelled", LabelledDirectory)


@pytest.fixture(autouse=True)
def probe_router(monkeypatch: pytest.MonkeyPatch) -> None:
    """One route on the mounting point, so a mount is visible while its routes are stubs."""
    router = APIRouter()

    @router.get("/probe")
    async def probe() -> dict[str, str]:
        return {"mounted": "yes"}

    monkeypatch.setattr(directory_api, "router", router)


@pytest.fixture(autouse=True)
def _no_config_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("OSPREY_CONFIG", str(tmp_path / "config.yml"))


def _config(
    control_system: dict[str, Any],
    section: dict[str, Any] | None,
    **extra: Any,
) -> dict[str, Any]:
    config: dict[str, Any] = {
        "channel_finder": {
            "pipeline_mode": "in_context",
            "pipelines": {"in_context": {"database": {"path": "/tmp/db.json", "type": "flat"}}},
        },
        "control_system": control_system,
    }
    if control_system == VA_BUILD:
        config["services"] = {"virtual_accelerator": {"port": 5064}}
    if section is not None:
        config["pv_directory"] = section
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


def _info(start, config: dict[str, Any]) -> dict[str, Any]:
    with start(config) as (_app, client):
        return client.get("/api/info").json()


def _directory_paths(app: FastAPI) -> list[str]:
    return [path for path in app.openapi()["paths"] if path.startswith("/api/directory")]


# -- absent, malformed ---------------------------------------------------------


def test_without_the_section_the_info_payload_is_unchanged(start) -> None:
    with start(_config(VA_BUILD, None)) as (app, client):
        info = client.get("/api/info").json()
        assert "directory" not in info
        assert _directory_paths(app) == []
        assert not hasattr(app.state, "directory_service")


def test_a_null_section_is_an_absent_one(start) -> None:
    config = _config(VA_BUILD, None)
    config["pv_directory"] = None
    with start(config) as (app, client):
        assert "directory" not in client.get("/api/info").json()
        assert _directory_paths(app) == []


def test_a_malformed_section_warns_and_serves_todays_panel(start, caplog) -> None:
    caplog.set_level(logging.WARNING, logger=_APP_LOGGER)
    with start(_config(LIVE_BUILD, {"type": "labelled"})) as (app, client):
        info = client.get("/api/info").json()
        assert "directory" not in info
        assert info["pipeline_type"] == "in_context"
        assert _directory_paths(app) == []
        assert client.get("/api/directory/probe").status_code == 404
    warnings = [r.getMessage() for r in caplog.records if r.name == _APP_LOGGER]
    assert any("pv_directory.settings.url" in message for message in warnings)


# -- configured ----------------------------------------------------------------


def test_a_configured_section_mounts_the_directory_routes(start) -> None:
    with start(_config(VA_BUILD, {})) as (app, client):
        assert _directory_paths(app) == ["/api/directory/probe"]
        assert client.get("/api/directory/probe").json() == {"mounted": "yes"}
        assert isinstance(app.state.directory_service, service.DirectoryService)


def test_the_descriptor_is_present_when_configured(start) -> None:
    info = _info(start, _config(VA_BUILD, {}))

    assert info["directory"] == {
        "label": "All PVs",
        "configured_services": [],
        "canopy": False,
        "ariel": False,
    }
    assert info["pipeline_type"] == "in_context"


def test_the_label_follows_a_target_move(start, record: _Record) -> None:
    config = _config(LIVE_BUILD, LABELLED)
    config["services"] = {"virtual_accelerator": {"port": 5064}}
    with start(config) as (_app, client):
        assert client.get("/api/info").json()["directory"]["label"] == "Facility PVs"
        record.move("va", 1)
        assert client.get("/api/info").json()["directory"]["label"] == "All PVs"
        record.move("live", 2)
        assert client.get("/api/info").json()["directory"]["label"] == "Facility PVs"


def test_a_va_only_section_at_live_is_labelled_all_pvs(start, record: _Record) -> None:
    record.move("live", 1)

    info = _info(start, _config(VA_BUILD, {}))

    assert info["directory"]["label"] == "All PVs"


def test_configured_services_follow_the_section(start, tmp_path: Path) -> None:
    replay = tmp_path / "alarms.ndjson"
    replay.write_text("", encoding="utf-8")
    section = {
        **LABELLED,
        "alarm_log": {"type": "alarm_logger", "settings": {"url": "http://alarms.example"}},
        "alarm_state": {"replay_file": "alarms.ndjson"},
        "put_log": {"type": "put_log", "settings": {"url": "http://puts.example"}},
    }

    info = _info(start, _config(LIVE_BUILD, section))

    assert info["directory"]["configured_services"] == ["alarm_log", "alarm_state", "put_log"]


def test_a_missing_replay_file_still_serves_the_directory(start) -> None:
    section = {"alarm_state": {"replay_file": "replays/not-there.ndjson"}}
    with start(_config(VA_BUILD, section)) as (app, client):
        info = client.get("/api/info").json()
        assert info["directory"]["configured_services"] == ["alarm_state"]
        assert _directory_paths(app) == ["/api/directory/probe"]


# -- canopy and ariel ----------------------------------------------------------


def test_canopy_says_the_panel_is_configured(start) -> None:
    on = _config(VA_BUILD, {}, web={"panels": {"canopy": {}}})
    off = _config(VA_BUILD, {}, web={"panels": {"canopy": {"enabled": False}}})

    assert _info(start, on)["directory"]["canopy"] is True
    assert _info(start, off)["directory"]["canopy"] is False


def test_ariel_needs_the_panel_and_a_section(start) -> None:
    section = {"database": {"url": "postgresql://ariel.example/logbook"}}
    selected = _config(VA_BUILD, {}, web={"panels": {"ariel": True}}, ariel=section)
    not_selected = _config(VA_BUILD, {}, web={"panels": {"canopy": True}}, ariel=section)
    no_section = _config(VA_BUILD, {}, web={"panels": {"ariel": True}})

    descriptor = _info(start, selected)["directory"]
    assert descriptor["ariel"] is True
    assert descriptor["configured_services"] == ["logbook"]
    assert _info(start, not_selected)["directory"]["ariel"] is False
    assert _info(start, no_section)["directory"]["ariel"] is False


# -- the floor -----------------------------------------------------------------


@pytest.fixture()
def floor() -> Iterator[None]:
    """The shim and the service re-imported with the PV directory modules unimportable."""
    with pytest.MonkeyPatch.context() as patch:
        for name in _NEW_MODULES:
            patch.setitem(sys.modules, name, None)
        importlib.reload(_connectors)
        importlib.reload(service)
        yield
    importlib.reload(_connectors)
    importlib.reload(service)


@pytest.mark.usefixtures("floor")
def test_on_the_floor_the_label_is_all_pvs(start) -> None:
    with start(_config(VA_BUILD, {})) as (app, client):
        assert client.get("/api/info").json()["directory"]["label"] == "All PVs"
        assert _directory_paths(app) == ["/api/directory/probe"]


# -- the shared routes fixture -------------------------------------------------


def _routes() -> dict[str, Any]:
    return json.loads(_FIXTURE.read_text(encoding="utf-8"))


def test_the_fixture_names_every_json_panel_route() -> None:
    assert set(_routes()) == {
        "search",
        "count",
        "pv/{name}",
        "ioc/{name}",
        "logbook/availability",
        "values",
    }


def test_the_fixture_search_is_a_directory_page() -> None:
    page = _routes()["search"]
    rows = [contract.DirectoryRow(**row) for row in page["rows"]]
    rebuilt = contract.DirectoryPage(
        **{**page, "source": contract.Source(**page["source"]), "rows": rows}
    )

    assert _connectors.as_payload(rebuilt) == page


def test_the_fixture_value_rows_carry_int_severity_and_iso_timestamps() -> None:
    values = _routes()["values"]

    assert [ValueRow.from_wire(row).to_wire() for row in values] == values
    found = [row for row in values if row["outcome"] == "found"]
    assert found
    for row in found:
        assert isinstance(row["severity"], int)
        assert datetime.fromisoformat(row["timestamp"]).tzinfo is not None
