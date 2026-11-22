"""The in-process simulator's journal carries each write's time, and replay keeps it.

Two connectors on one simulator view share the writes journal. A status that
answers its command for a stated time (``responds``) runs from the time its
command was written, in every process: replaying the journal neither restarts
a response nor revives one that ended.
"""

from __future__ import annotations

import functools
import json
from pathlib import Path
from typing import Any

import pytest

from osprey_connectors.control_system._va_in_process_connector import (
    JOURNAL_DIR,
    JOURNAL_FILE,
    VAInProcessConnector,
    active_set_sha256,
    simulation_state_dir,
)
from osprey_connectors.simulation import composite as composite_module
from osprey_connectors.simulation.view import SCHEMAS
from osprey_connectors_tests._write_fakes import arm_switch, writes_enabled_config

T0 = 1_760_000_000.0
FOR_S = 2.0

MOVE = "T:STAGE:MOVE:SP"
STATUS = "T:STAGE:STATUS"
POSITION_SP = "T:STAGE:POSITION:SP"
POSITION_RB = "T:STAGE:POSITION:RB"
HEAT_SP = "T:HEAT:SP"
LABELS = ("DONE", "MOVING")
DONE, MOVING = (LABELS.index(label) for label in LABELS)


def _channel(address: str, **fields: Any) -> dict[str, Any]:
    role = fields.pop("role", "readback")
    return {
        "address": address,
        "role": role,
        "pair": fields.pop("pair", address if role == "setpoint" else None),
        "value_type": fields.pop("value_type", "float"),
        "unit": None,
        "description": None,
        "writable": role == "setpoint",
        "value_range": None,
        "owner": "texture",
        "on": None,
        **fields,
    }


def render(root: Path, *, for_s: float = FOR_S, tick_s: float | None = None) -> Path:
    """A render whose stage has a move command, a timed status and a started position.

    Returns the simulator view's directory; ``config.yml`` beside it states
    ``tick_s`` when given.
    """
    channels = [
        _channel(MOVE, role="setpoint", value_type="int"),
        _channel(STATUS, value_type="enum", options=list(LABELS)),
        _channel(POSITION_SP, role="setpoint", pair=POSITION_RB, started_by=MOVE),
        _channel(POSITION_RB),
        _channel(HEAT_SP, role="setpoint"),
    ]
    documents = {
        "served_models.json": {"models": ["texture"]},
        "addresses.json": {"channels": sorted(c["address"] for c in channels), "status": []},
        "variables.json": {
            "code": "T",
            "models": [
                {
                    "name": "texture",
                    "engine": "texture",
                    "served": True,
                    "settings": {},
                    "deck": None,
                    "wiring": [],
                }
            ],
            "channels": channels,
        },
        "seeds.json": {
            "seeds": {
                STATUS: {
                    "nominal": "DONE",
                    "responds": {"to": MOVE, "for_s": for_s, "value": "MOVING"},
                },
                POSITION_SP: {"nominal": 1.0},
                HEAT_SP: {"nominal": 20.0},
            }
        },
        "scenarios.json": {"scenarios": [{"name": "nominal"}]},
    }
    view = root / "data" / "simulator"
    view.mkdir(parents=True, exist_ok=True)
    for name, document in documents.items():
        (view / name).write_text(
            json.dumps({"schema": SCHEMAS[name], **document}), encoding="utf-8"
        )
    config: dict[str, Any] = {"project_root": str(root)}
    if tick_s is not None:
        config["simulation"] = {"tick_s": tick_s}
    (root / "config.yml").write_text(json.dumps(config), encoding="utf-8")
    return view


@pytest.fixture
def writes_on(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    arm_switch(monkeypatch, tmp_path / "armed-data")
    monkeypatch.setattr("osprey_connectors.config.get_config_value", writes_enabled_config)


class Clock:
    """An injected clock at ``now``, stepping by ``step`` after each reading."""

    def __init__(self) -> None:
        self.now = T0
        self.step = 0.0

    def __call__(self) -> float:
        reading = self.now
        self.now += self.step
        return reading


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> Clock:
    """The one clock every connector's simulator is built with, at ``T0``."""
    injected = Clock()
    monkeypatch.setattr(
        composite_module, "Composite", functools.partial(composite_module.Composite, clock=injected)
    )
    return injected


async def _connected(view: Path, clock: Clock) -> VAInProcessConnector:  # noqa: ARG001 - built with it
    connector = VAInProcessConnector()
    await connector.connect({"simulator_view": str(view), "response_delay_ms": 0})
    return connector


@pytest.fixture
async def two(
    tmp_path: Path,
    writes_on: None,  # noqa: ARG001 - a fixture, used for its effect
    clock: Clock,
) -> Any:
    """Two connectors serving one render, as two processes would."""
    view = render(tmp_path)
    first, second = await _connected(view, clock), await _connected(view, clock)
    yield first, second
    await first.disconnect()
    await second.disconnect()


async def _read(connector: VAInProcessConnector, address: str) -> object:
    return (await connector.read_channel(address)).value


async def test_a_second_process_sees_a_running_response(two: Any, clock: Clock) -> None:
    first, second = two

    await first.write_channel(MOVE, 1)
    clock.now = T0 + FOR_S / 2

    assert await _read(second, STATUS) == MOVING
    clock.now = T0 + FOR_S
    assert await _read(second, STATUS) == DONE


async def test_an_unrelated_write_from_another_process_does_not_restart_a_response(
    two: Any, clock: Clock
) -> None:
    first, second = two
    await first.write_channel(MOVE, 1)
    clock.now = T0 + FOR_S - 0.5

    await second.write_channel(HEAT_SP, 30.0)

    assert await _read(first, STATUS) == MOVING
    clock.now = T0 + FOR_S
    assert await _read(first, STATUS) == DONE
    assert await _read(first, HEAT_SP) == pytest.approx(30.0)


@pytest.mark.usefixtures("writes_on")
async def test_a_response_that_ended_before_the_replay_stays_ended(
    tmp_path: Path, clock: Clock
) -> None:
    view = render(tmp_path)
    first = await _connected(view, clock)
    await first.write_channel(MOVE, 1)
    clock.now = T0 + FOR_S + 1.0

    late = await _connected(view, clock)

    assert await _read(late, STATUS) == DONE
    assert await _read(late, MOVE) == 1
    await first.disconnect()
    await late.disconnect()


def _plant(view: Path, connector: VAInProcessConnector, writes: list[list[object]]) -> None:
    """Write the journal as an older process would, with ``writes`` as its entries."""
    assert connector._composite is not None
    journal = simulation_state_dir(view) / JOURNAL_DIR / JOURNAL_FILE
    journal.parent.mkdir(parents=True, exist_ok=True)
    journal.write_text(
        json.dumps(
            {
                "active_set_sha256": active_set_sha256(connector._composite.active),
                "seq": len(writes),
                "writes": writes,
            }
        ),
        encoding="utf-8",
    )


async def test_a_three_element_entry_replays_its_value_and_fires_nothing(
    tmp_path: Path, clock: Clock
) -> None:
    view = render(tmp_path)
    connector = await _connected(view, clock)

    _plant(view, connector, [[1, POSITION_SP, 4.0], [2, MOVE, 1]])

    assert await _read(connector, POSITION_RB) == pytest.approx(4.0)
    assert await _read(connector, MOVE) == 1
    assert await _read(connector, STATUS) == DONE
    await connector.disconnect()


async def test_a_four_element_entry_fires_at_its_time(tmp_path: Path, clock: Clock) -> None:
    view = render(tmp_path)
    connector = await _connected(view, clock)

    _plant(view, connector, [[1, MOVE, 1, T0 - FOR_S / 2]])

    assert await _read(connector, STATUS) == MOVING
    clock.now = T0 + FOR_S / 2
    assert await _read(connector, STATUS) == DONE
    await connector.disconnect()


@pytest.mark.parametrize("t_s", ["soon", True, float("nan")], ids=["text", "bool", "nan"])
async def test_an_entry_whose_time_is_no_number_rejects_the_journal(
    tmp_path: Path, clock: Clock, t_s: object
) -> None:
    view = render(tmp_path)
    connector = await _connected(view, clock)

    _plant(view, connector, [[1, HEAT_SP, 30.0, t_s]])

    assert await _read(connector, HEAT_SP) == pytest.approx(20.0)
    await connector.disconnect()


async def test_replay_releases_a_target_only_when_its_command_follows_it(two: Any) -> None:
    first, second = two

    await first.write_channel(MOVE, 1)
    await first.write_channel(POSITION_SP, 4.0)

    assert await _read(second, POSITION_SP) == pytest.approx(4.0)
    assert await _read(second, POSITION_RB) == pytest.approx(1.0)
    await first.write_channel(MOVE, 1)
    assert await _read(second, POSITION_RB) == pytest.approx(4.0)


@pytest.mark.usefixtures("writes_on")
async def test_a_write_journals_its_time(tmp_path: Path, clock: Clock) -> None:
    view = render(tmp_path)
    connector = await _connected(view, clock)

    await connector.write_channel(HEAT_SP, 30.0)

    journal = json.loads(
        (simulation_state_dir(view) / JOURNAL_DIR / JOURNAL_FILE).read_text(encoding="utf-8")
    )
    assert journal["writes"] == [[1, HEAT_SP, 30.0, T0]]
    await connector.disconnect()


async def test_every_process_ends_a_response_at_the_instant_the_journal_carries(
    two: Any, clock: Clock, tmp_path: Path
) -> None:
    first, second = two
    clock.step = 0.25

    await first.write_channel(MOVE, 1, confirm=False)

    clock.step = 0.0
    journal = json.loads(
        (
            simulation_state_dir(tmp_path / "data" / "simulator") / JOURNAL_DIR / JOURNAL_FILE
        ).read_text(encoding="utf-8")
    )
    written = journal["writes"][-1][3]
    clock.now = written + FOR_S - 0.01
    assert [await _read(first, STATUS), await _read(second, STATUS)] == [MOVING, MOVING]
    clock.now = written + FOR_S
    assert [await _read(first, STATUS), await _read(second, STATUS)] == [DONE, DONE]
