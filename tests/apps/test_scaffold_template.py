"""The template `osprey app new` copies: a valid application that names no channel of its own."""

from __future__ import annotations

import os
import shutil
import zipfile
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.middleware import Middleware

import osprey.apps
from osprey.apps.boundary import check_package
from osprey.apps.loader import load
from osprey.apps.sdk import LAUNCH_TOKEN_HEADER, OWNER_STATE_KEY
from osprey.interfaces.common_middleware import WebAuthMiddleware
from osprey.mcp_server.startup import application_server
from tests.test_offline_build_env import NO_ROUTE

TEMPLATE = Path(osprey.apps.__file__).resolve().parent / "scaffold-app"
APP_NAME = "scaffold-app"
PACKAGE = TEMPLATE / "scaffold_app"
TOKEN = "launch-token-value"
OWNER = "alice"
CHANNEL = "DEMO:CHANNEL:SP"
CHANNEL_SETTING = "apps.scaffold-app.config.channel"


@pytest.fixture
def scaffold(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """The template's package, imported from its source tree."""
    monkeypatch.syspath_prepend(str(TEMPLATE))
    import scaffold_app

    assert Path(scaffold_app.__file__).resolve().parent == PACKAGE
    return scaffold_app


class StandInConnector:
    """Holds one value per channel, and records what is written."""

    def __init__(self) -> None:
        self.values: dict[str, float] = {CHANNEL: 1.5}
        self.reads: list[str] = []

    async def read_channel(self, channel: str) -> SimpleNamespace:
        self.reads.append(channel)
        return SimpleNamespace(value=self.values[channel])

    async def write_channel_checked(self, channel: str, value: float) -> SimpleNamespace:
        self.values[channel] = value
        return SimpleNamespace(outcome="confirmed")

    async def disconnect(self) -> None:
        return None


class _GatedAs:
    """Stamp the owner the gate would have stamped."""

    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if scope["type"] == "http":
            scope.setdefault("state", {})[OWNER_STATE_KEY] = OWNER
        await self.app(scope, receive, send)


def _client(
    monkeypatch: pytest.MonkeyPatch,
    connector: StandInConnector,
    config: dict[str, Any] | None,
) -> TestClient:
    from scaffold_app import server as backend

    monkeypatch.setenv(backend.LAUNCH_TOKEN_ENV, TOKEN)
    monkeypatch.setattr(backend, "_section", lambda: {"type": "virtual_accelerator"})

    async def connector_for(_target: str) -> StandInConnector:
        return connector

    app: FastAPI = backend.create_app(connector_for, config)
    app.user_middleware = [
        entry for entry in app.user_middleware if entry.cls is not WebAuthMiddleware
    ] + [Middleware(_GatedAs)]
    return TestClient(app)


def _write(client: TestClient, scaffold: ModuleType) -> Any:
    body = {"value": 2.5, "target": "va", "readback": 1.5}
    return client.post(scaffold.WRITE_ROUTE, json=body, headers={LAUNCH_TOKEN_HEADER: TOKEN})


def test_the_loader_reads_the_template_as_an_application(tmp_path: Path) -> None:
    shutil.copytree(TEMPLATE, tmp_path / "apps" / APP_NAME)

    [manifest] = load(tmp_path)

    assert manifest.name == APP_NAME
    assert list(manifest.containers) == [APP_NAME]
    assert manifest.panel is not None
    assert manifest.panel.launch_token
    assert ("GET", "/api/state") in [(r.method, r.path) for r in manifest.panel.agent_routes]
    assert manifest.tools is not None
    assert manifest.tools.module == "scaffold_app.tools"
    assert len(manifest.tools.read) == 1
    [approval] = manifest.tools.approval
    assert approval.state_path == "/api/state"
    assert {"approved_target", "approved_readback"} <= set(approval.require.all_of)
    assert manifest.vendor == []


def test_the_package_stays_inside_the_boundary() -> None:
    assert sorted(path.name for path in PACKAGE.glob("*.py")) == [
        "__init__.py",
        "server.py",
        "tools.py",
    ]
    assert check_package(PACKAGE, app=APP_NAME) == []


def test_the_template_names_no_channel_of_its_own() -> None:
    for path in PACKAGE.glob("*.py"):
        assert "SR:" not in path.read_text(), path


@pytest.mark.usefixtures("scaffold")
def test_the_tool_server_registers_exactly_the_declared_tools(tmp_path: Path) -> None:
    import asyncio

    from scaffold_app.tools import register

    shutil.copytree(TEMPLATE, tmp_path / "apps" / APP_NAME)
    [manifest] = load(tmp_path)
    assert manifest.tools is not None
    declared = {*manifest.tools.read, *(entry.tool for entry in manifest.tools.approval)}
    server = application_server(APP_NAME)
    register(server)
    assert {tool.name for tool in asyncio.run(server.list_tools())} == declared


def test_with_no_channel_the_state_names_the_config_key(
    scaffold: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    connector = StandInConnector()
    with _client(monkeypatch, connector, {}) as client:
        state = client.get(scaffold.STATE_ROUTE)

    assert state.status_code == 200
    assert CHANNEL_SETTING in state.json()["error"]
    assert state.json()["channel"] is None
    assert connector.reads == []


def test_with_no_channel_the_write_is_refused_with_409(
    scaffold: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    connector = StandInConnector()
    with _client(monkeypatch, connector, {}) as client:
        answer = _write(client, scaffold)

    assert answer.status_code == 409
    assert CHANNEL_SETTING in answer.text
    assert connector.values == {CHANNEL: 1.5}


def test_the_channel_comes_from_the_rendered_app_config(
    scaffold: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from scaffold_app import server as backend

    config = tmp_path / "config.yml"
    config.write_text(f"channel: {CHANNEL}\n", encoding="utf-8")
    monkeypatch.setenv(backend.APP_CONFIG_ENV, str(config))
    connector = StandInConnector()
    with _client(monkeypatch, connector, None) as client:
        state = client.get(scaffold.STATE_ROUTE).json()
        written = _write(client, scaffold)

    assert state == {"channel": CHANNEL, "target": "va", "readback": 1.5, "error": None}
    assert written.status_code == 200, written.text
    assert connector.values[CHANNEL] == 2.5


@pytest.mark.usefixtures("scaffold")
def test_an_absent_app_config_is_no_channel(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from scaffold_app import server as backend

    monkeypatch.setenv(backend.APP_CONFIG_ENV, str(tmp_path / "missing.yml"))
    assert backend.configured_channel(backend.read_app_config()) is None
    monkeypatch.delenv(backend.APP_CONFIG_ENV)
    assert backend.read_app_config() == {}


@pytest.mark.slow
def test_a_build_with_no_route_to_an_index_makes_the_wheel(
    offline_build_env: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from osprey.deployment.wheel_build import _build_wheel

    for name, value in {**offline_build_env, **NO_ROUTE}.items():
        monkeypatch.setitem(os.environ, name, value)
    source = tmp_path / "source"
    shutil.copytree(TEMPLATE, source, ignore=shutil.ignore_patterns("__pycache__"))
    out = tmp_path / "out"
    out.mkdir()

    wheel = _build_wheel(source, out)

    assert wheel.name.startswith("scaffold_app-")
    names = zipfile.ZipFile(wheel).namelist()
    assert {
        "scaffold_app/__init__.py",
        "scaffold_app/server.py",
        "scaffold_app/tools.py",
        "scaffold_app/panel/index.html",
        "scaffold_app/panel/panel.js",
        "scaffold_app/panel/panel.css",
    } <= set(names)
    assert not [name for name in names if "/tests/" in name or "__pycache__" in name]


#: A stand-in baseline beside the live machine: the container's name server is the stand-in's.
SWITCHABLE_SECTION = {
    "type": "live_standin",
    "connector": {
        "live_standin": {"gateways": {"read_only": {"address": "localhost", "port": 19090}}},
        "epics": {"gateways": {"read_only": {"address": "cagw", "port": 5064}}, "timeout_s": 3.0},
    },
}


@pytest.fixture
def built(scaffold: ModuleType, monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """What ``build_connector`` hands the factory, in a fresh process on ``SWITCHABLE_SECTION``."""
    import copy
    import importlib

    backend = importlib.import_module(f"{scaffold.__name__}.server")

    seen: list[dict[str, Any]] = []

    async def create(config: dict[str, Any], control_target: str) -> SimpleNamespace:
        if control_target == "unreachable":
            raise OSError("no route")
        seen.append({"config": config, "target": control_target})
        return SimpleNamespace()

    monkeypatch.setattr(backend, "_ca_target", None)
    monkeypatch.setattr(backend, "_section", lambda: copy.deepcopy(SWITCHABLE_SECTION))
    monkeypatch.setattr(backend, "register_builtin_connectors", lambda: None)
    monkeypatch.setattr(
        backend.ConnectorFactory, "create_control_system_connector", staticmethod(create)
    )
    return seen


@pytest.mark.parametrize(
    ("target", "kind", "gateways"),
    [("standin", "live_standin", False), ("live", "epics", True)],
)
def test_in_a_container_only_the_baseline_target_drops_its_gateways(
    monkeypatch: pytest.MonkeyPatch,
    built: list[dict[str, Any]],
    target: str,
    kind: str,
    gateways: bool,
) -> None:
    """The container's name server is the baseline's; any other target keeps its own route."""
    import asyncio

    from scaffold_app import server as backend

    monkeypatch.setenv("EPICS_CA_NAME_SERVERS", "live-standin:19090")

    asyncio.run(backend.build_connector(target))

    [call] = built
    assert call["target"] == target
    assert call["config"]["type"] == kind
    assert ("gateways" in call["config"]["connector"][kind]) is gateways


def test_a_second_channel_access_target_in_one_process_is_refused(
    built: list[dict[str, Any]],
) -> None:
    """One context per process: building it for a second machine would move the first."""
    import asyncio

    from scaffold_app import server as backend

    asyncio.run(backend.build_connector("live"))
    asyncio.run(backend.build_connector("live"))
    with pytest.raises(ValueError, match="serves live.*to serve standin"):
        asyncio.run(backend.build_connector("standin"))

    assert [call["target"] for call in built] == ["live", "live"]


def test_a_failed_connect_leaves_channel_access_unclaimed(
    monkeypatch: pytest.MonkeyPatch, built: list[dict[str, Any]]
) -> None:
    import asyncio

    from scaffold_app import server as backend

    monkeypatch.setattr(backend, "resolve_target", lambda _section, _target: "epics")
    with pytest.raises(OSError, match="no route"):
        asyncio.run(backend.build_connector("unreachable"))
    asyncio.run(backend.build_connector("standin"))

    assert [call["target"] for call in built] == ["standin"]
