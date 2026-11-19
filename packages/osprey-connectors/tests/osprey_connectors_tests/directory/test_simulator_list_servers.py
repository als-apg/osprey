"""The simulator directory lists every simulated server, from the view alone."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from osprey_connectors.directory import DirectoryQuery, ServerRow
from osprey_connectors.directory._simulator import SimulatorDirectory
from osprey_connectors.service_answers import BackendSpec, Unavailable
from osprey_connectors.simulation.servers import TEXTURE_SERVER
from osprey_connectors.simulation.view import (
    ADDRESSES_FILE,
    ADDRESSES_SCHEMA,
    IOC_HEALTH_SIGNALS,
    SCENARIOS_FILE,
    SCENARIOS_SCHEMA,
    SERVED_MODELS_FILE,
    SERVED_MODELS_SCHEMA,
    VARIABLES_FILE,
    VARIABLES_SCHEMA,
    VIEW_RELPATH,
)

DEADLINE = 6.0
HOST = "virtual-accelerator"
HEALTH = len(IOC_HEALTH_SIGNALS)


def _channel(address: str, owner: str) -> dict[str, Any]:
    return {
        "address": address,
        "role": "readback",
        "pair": None,
        "value_type": "float",
        "unit": None,
        "description": None,
        "writable": False,
        "value_range": None,
        "owner": owner,
        "on": None,
    }


def _health(model: str) -> list[str]:
    return [f"T:SIM:{model}:{signal}" for signal in IOC_HEALTH_SIGNALS]


_CHANNELS = [
    _channel("T:SR:QF1:SP", "SR"),
    _channel("T:SR:QF1:RB", "SR"),
    _channel("T:RF:CW:FLOW", "texture"),
    _channel("T:RF:CW:ENABLE", "texture"),
    _channel("T:LN:GUN:SP", "LN"),
]


def _write_view(
    project: Path,
    *,
    channels: list[dict[str, Any]],
    ioc_health: list[str] | None = None,
    stopped: list[str] | None = None,
) -> None:
    """A view serving physics models ``SR`` and ``BARE`` (no deck) and the texture.

    ``LN`` is in the facility file and not served, so its channels are the texture's.
    """
    addresses: dict[str, Any] = {
        "schema": ADDRESSES_SCHEMA,
        "channels": sorted(entry["address"] for entry in channels),
        "status": ["T:SIM:BARE:STATUS", "T:SIM:SR:STATUS"],
    }
    if ioc_health is not None:
        addresses["ioc_health"] = ioc_health
    documents: dict[str, Any] = {
        ADDRESSES_FILE: addresses,
        SERVED_MODELS_FILE: {"schema": SERVED_MODELS_SCHEMA, "models": ["BARE", "SR", "texture"]},
        VARIABLES_FILE: {
            "schema": VARIABLES_SCHEMA,
            "code": "T",
            "models": [
                {"name": "BARE", "engine": "pyat", "served": True},
                {"name": "LN", "engine": "pyat", "served": False},
                {"name": "SR", "engine": "pyat", "served": True},
                {"name": "texture", "engine": "texture", "served": True},
            ],
            "channels": channels,
        },
    }
    if stopped is not None:
        documents[SCENARIOS_FILE] = {
            "schema": SCENARIOS_SCHEMA,
            "scenarios": [{"name": "nominal"}, {"name": "outage", "stopped_servers": stopped}],
        }
    view_dir = project / VIEW_RELPATH
    view_dir.mkdir(parents=True)
    for name, document in documents.items():
        (view_dir / name).write_text(json.dumps(document), encoding="utf-8")


def _directory(project: Path | None) -> SimulatorDirectory:
    spec = BackendSpec(
        kind="directory", type="simulator_view", target="va", settings={}, rehearsal=False
    )
    return SimulatorDirectory(spec, project_dir=project)


async def test_every_simulated_server_is_one_row_with_its_channel_count(tmp_path: Path) -> None:
    _write_view(
        tmp_path,
        channels=_CHANNELS,
        ioc_health=_health("BARE") + _health("SR") + _health("texture"),
    )

    servers = await _directory(tmp_path).list_servers(DEADLINE)

    assert servers == [
        ServerRow(name="sim-BARE", host=HOST, channel_count=1 + HEALTH, inactive_count=0),
        ServerRow(name="sim-SR", host=HOST, channel_count=2 + 1 + HEALTH, inactive_count=0),
        ServerRow(name=TEXTURE_SERVER, host=HOST, channel_count=3 + HEALTH, inactive_count=0),
    ]


async def test_the_counts_agree_with_the_directorys_own_search(tmp_path: Path) -> None:
    _write_view(tmp_path, channels=_CHANNELS, ioc_health=_health("SR") + _health("texture"))
    directory = _directory(tmp_path)

    servers = await directory.list_servers(DEADLINE)

    assert not isinstance(servers, Unavailable)
    for server in servers:
        query = DirectoryQuery(server=[server.name])
        assert await directory.count(query, DEADLINE) == server.channel_count
    assert sum(server.channel_count for server in servers) == await directory.count(
        DirectoryQuery(), DEADLINE
    )


async def test_a_server_serving_no_channel_is_still_listed(tmp_path: Path) -> None:
    _write_view(tmp_path, channels=[_channel("T:SR:QF1:RB", "SR")])

    servers = await _directory(tmp_path).list_servers(DEADLINE)

    assert not isinstance(servers, Unavailable)
    assert {server.name: server.channel_count for server in servers} == {
        "sim-BARE": 1,
        "sim-SR": 2,
        TEXTURE_SERVER: 0,
    }


async def test_a_stopped_server_is_still_listed(tmp_path: Path) -> None:
    _write_view(tmp_path, channels=_CHANNELS, stopped=["sim-SR", TEXTURE_SERVER])

    servers = await _directory(tmp_path).list_servers(DEADLINE)

    assert not isinstance(servers, Unavailable)
    assert {server.name: server.channel_count for server in servers} == {
        "sim-BARE": 1,
        "sim-SR": 3,
        TEXTURE_SERVER: 3,
    }


async def test_rows_are_unique_and_sorted_by_name(tmp_path: Path) -> None:
    _write_view(tmp_path, channels=_CHANNELS, ioc_health=_health("SR"))

    servers = await _directory(tmp_path).list_servers(DEADLINE)

    assert not isinstance(servers, Unavailable)
    names = [server.name for server in servers]
    assert names == sorted(set(names))


async def test_a_missing_view_is_unavailable_never_an_empty_list(tmp_path: Path) -> None:
    servers = await _directory(tmp_path).list_servers(DEADLINE)

    assert isinstance(servers, Unavailable)
    assert servers.reason


async def test_no_project_directory_is_unavailable() -> None:
    servers = await _directory(None).list_servers(DEADLINE)

    assert isinstance(servers, Unavailable)
    assert "project directory" in servers.reason
