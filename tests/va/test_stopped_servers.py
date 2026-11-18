"""A scenario's ``stopped_servers`` in process: the composite and the in-process connector.

The served half -- held Channel Access and PVAccess clients across a stop --
is ``TestStoppedServer`` in ``tests/va/test_lume_pva_seam.py``, run by the
live-CA gate.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest

from osprey_connectors.control_system._va_in_process_connector import (
    VAInProcessConnector,
    simulation_state_dir,
)
from osprey_connectors.errors import ChannelUnreachableError
from osprey_connectors.simulation.composite import Composite
from osprey_connectors.simulation.servers import TEXTURE_SERVER, served_servers, server_addresses
from osprey_connectors.simulation.view import SCHEMAS, SimulatorView

STATUS = "T:SIM:M:STATUS"
SETPOINT = "T:SP"
HEARTBEAT = "T:SIM:M:HEARTBEAT"
TEXTURE_HEARTBEAT = "T:SIM:texture:HEARTBEAT"
STOP_M = "m-stopped"
STOP_TEXTURE = "texture-stopped"


def _view(path: Path) -> Path:
    """Model ``M`` (no deck) and the texture, with a scenario stopping each one's IOC."""
    documents = {
        "served_models.json": {"models": ["M", "texture"]},
        "addresses.json": {
            "channels": [SETPOINT],
            "status": [STATUS],
            "ioc_health": [HEARTBEAT, TEXTURE_HEARTBEAT],
        },
        "variables.json": {
            "code": "T",
            "models": [
                {
                    "name": name,
                    "engine": engine,
                    "served": True,
                    "settings": {},
                    "deck": None,
                    "wiring": [],
                }
                for name, engine in (("M", "stub"), ("texture", "texture"))
            ],
            "channels": [
                {
                    "address": SETPOINT,
                    "owner": "texture",
                    "role": "setpoint",
                    "value_type": "float",
                    "writable": True,
                    "pair": None,
                    "unit": None,
                    "description": None,
                    "value_range": None,
                    "on": None,
                }
            ],
        },
        "seeds.json": {"seeds": {SETPOINT: {"nominal": 5.0}}},
        "scenarios.json": {
            "scenarios": [
                {"name": "nominal"},
                {"name": STOP_M, "stopped_servers": ["sim-M"]},
                {"name": STOP_TEXTURE, "stopped_servers": [TEXTURE_SERVER]},
                {"name": "nothing-stopped", "stopped_servers": ["sim-nothing"]},
            ]
        },
    }
    view = path / "simulator"
    view.mkdir(parents=True, exist_ok=True)
    for name, document in documents.items():
        (view / name).write_text(
            json.dumps({"schema": SCHEMAS[name], **document}), encoding="utf-8"
        )
    return view


def _activate(state_dir: Path, *names: str) -> None:
    """Write the active set, moving the file's modification time forward."""
    state_dir.mkdir(parents=True, exist_ok=True)
    path = state_dir / "active_scenarios"
    before = path.stat().st_mtime_ns if path.exists() else 0
    path.write_text("".join(f"{name}\n" for name in names), encoding="utf-8")
    stamp = max(before + 1_000_000, path.stat().st_mtime_ns)
    os.utime(path, ns=(stamp, stamp))


def test_each_address_belongs_to_the_server_the_directory_names(tmp_path: Path) -> None:
    view = SimulatorView.open(_view(tmp_path))
    owners = {address: view.channel(address).owner for address in view.channels()}
    physics = [model.name for model in view.physics_models()]
    assert server_addresses(
        physics, owners, view.status_addresses(), view.ioc_health_by_model()
    ) == {
        "sim-M": frozenset({STATUS, HEARTBEAT}),
        TEXTURE_SERVER: frozenset({SETPOINT, TEXTURE_HEARTBEAT}),
    }
    assert served_servers(["M"]) == {"sim-M", TEXTURE_SERVER}


@pytest.mark.parametrize(
    ("active", "stopped"),
    [
        ((), frozenset()),
        ((STOP_M,), frozenset({STATUS, HEARTBEAT})),
        ((STOP_TEXTURE,), frozenset({SETPOINT, TEXTURE_HEARTBEAT})),
        (("nothing-stopped",), frozenset()),
    ],
)
def test_the_composite_names_the_stopped_servers_addresses(
    tmp_path: Path, active: tuple[str, ...], stopped: frozenset[str]
) -> None:
    _activate(tmp_path / "state", *active)
    composite = Composite(_view(tmp_path), state_dir=tmp_path / "state", model_log=False)
    assert composite.stopped() == stopped
    assert not composite.moving() & stopped


def test_a_stop_follows_the_active_set_without_a_restart(tmp_path: Path) -> None:
    composite = Composite(_view(tmp_path), state_dir=tmp_path / "state", model_log=False)
    assert composite.stopped() == frozenset()
    _activate(tmp_path / "state", STOP_M)
    assert composite.stopped() == {STATUS, HEARTBEAT}
    _activate(tmp_path / "state")
    assert composite.stopped() == frozenset()


@pytest.fixture
async def connected(tmp_path: Path) -> Any:
    view = _view(tmp_path)
    connector = VAInProcessConnector()
    await connector.connect({"simulator_view": str(view), "response_delay_ms": 0})
    yield connector, simulation_state_dir(view)
    await connector.disconnect()


async def test_a_stopped_servers_channel_gives_no_answer_then_answers_again(connected: Any) -> None:
    connector, state_dir = connected
    _activate(state_dir, STOP_M)
    for address in (STATUS, HEARTBEAT):
        with pytest.raises(ChannelUnreachableError):
            await connector.read_channel(address)
        with pytest.raises(ChannelUnreachableError):
            await connector.get_metadata(address)
    # The other IOC keeps answering.
    assert (await connector.read_channel(SETPOINT)).value == pytest.approx(5.0)
    assert (await connector.read_channel(TEXTURE_HEARTBEAT)).value is not None
    _activate(state_dir)
    assert (await connector.read_channel(HEARTBEAT)).value is not None
    assert (await connector.read_channel(STATUS)).value is not None


async def test_a_stopped_channels_subscription_stays_quiet(connected: Any) -> None:
    connector, state_dir = connected
    seen: list[Any] = []
    await connector.subscribe(HEARTBEAT, seen.append)
    _activate(state_dir, STOP_M)
    await connector._notify()
    assert seen == []


async def test_a_subscription_made_while_stopped_waits_and_is_told_disconnected(
    connected: Any,
) -> None:
    connector, state_dir = connected
    _activate(state_dir, STOP_M)
    seen: list[Any] = []
    states: list[bool] = []
    await connector.subscribe(HEARTBEAT, seen.append, on_connection=states.append)
    assert states == [False]
    await connector._notify()
    assert seen == [] and states == [False]
    _activate(state_dir)
    connector._sync()
    await connector._notify()
    assert states == [False, True]
    assert len(seen) == 1 and seen[0].value is not None


async def test_a_held_subscription_is_told_of_the_stop_and_the_restart(connected: Any) -> None:
    connector, state_dir = connected
    states: list[bool] = []
    await connector.subscribe(SETPOINT, lambda _: None, on_connection=states.append)
    assert states == [True]
    _activate(state_dir, STOP_TEXTURE)
    connector._sync()
    await connector._notify()
    await connector._notify()
    assert states == [True, False]
    _activate(state_dir)
    connector._sync()
    await connector._notify()
    assert states == [True, False, True]


async def test_a_restarted_channel_answers_its_current_value_not_one_held_at_the_stop(
    connected: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    connector, state_dir = connected
    composite = connector._composite
    now = [composite._started_s + 10.0]
    monkeypatch.setattr(composite, "_clock", lambda: now[0])
    seen: list[Any] = []
    await connector.subscribe(HEARTBEAT, seen.append)
    assert (await connector.read_channel(HEARTBEAT)).value == 10
    _activate(state_dir, STOP_M)
    with pytest.raises(ChannelUnreachableError):
        await connector.read_channel(HEARTBEAT)
    await connector._notify()
    now[0] += 90.0
    _activate(state_dir)
    assert (await connector.read_channel(HEARTBEAT)).value == 100
    seen.clear()
    await connector._notify()
    assert [reading.value for reading in seen] == [100]


async def test_a_write_to_a_stopped_servers_setpoint_sends_nothing(
    connected: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from osprey_connectors.control_system._base import WriteOutcome
    from osprey_connectors_tests._write_fakes import arm_switch, writes_enabled_config

    arm_switch(monkeypatch, tmp_path / "armed-data")
    monkeypatch.setattr("osprey_connectors.config.get_config_value", writes_enabled_config)
    connector, state_dir = connected
    _activate(state_dir, STOP_TEXTURE)
    result = await connector.write_channel(SETPOINT, 7.0, confirm=False)
    assert result.outcome is WriteOutcome.FAILED
    assert "no value was sent" in (result.error_message or "")
    _activate(state_dir)
    assert (await connector.read_channel(SETPOINT)).value == pytest.approx(5.0)
    # Running, the same write lands: the stop, not the policy, sent nothing.
    result = await connector.write_channel(SETPOINT, 7.0, confirm=False)
    assert result.outcome is WriteOutcome.UNREQUESTED
    assert (await connector.read_channel(SETPOINT)).value == pytest.approx(7.0)
