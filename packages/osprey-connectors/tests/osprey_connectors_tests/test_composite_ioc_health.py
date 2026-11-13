"""The composite and the in-process connector serve the view's IOC health addresses."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from lume.exceptions import ReadOnlyError
from lume.variables import IntVariable, ScalarVariable, StrVariable

from osprey_connectors.control_system._base import WriteOutcome
from osprey_connectors.control_system._va_in_process_connector import VAInProcessConnector
from osprey_connectors.simulation.composite import Composite
from osprey_connectors.simulation.status import STATUS_NO_DECK
from osprey_connectors.simulation.view import SCHEMAS, ViewSchemaError

STATUS = "T:SIM:M:STATUS"
SETPOINT = "T:SP"
HEARTBEAT = "T:SIM:M:HEARTBEAT"
UPTIME = "T:SIM:M:UPTIME"
ENGINEER = "T:SIM:M:ENGINEER"
TEXTURE_HEALTH = ("T:SIM:texture:HEARTBEAT", "T:SIM:texture:UPTIME", "T:SIM:texture:ENGINEER")
T0 = 1_760_000_000.0


class _Clock:
    def __init__(self) -> None:
        self.now = T0

    def __call__(self) -> float:
        return self.now


def _view(path: Path, health: tuple[str, ...] | None = (HEARTBEAT, UPTIME, ENGINEER)) -> Path:
    """A view whose physics model ``M`` has no deck, so it is served unbuilt."""
    addresses: dict[str, Any] = {"channels": [SETPOINT], "status": [STATUS]}
    if health is not None:
        addresses["ioc_health"] = [*health, *TEXTURE_HEALTH]
    documents = {
        "served_models.json": {"models": ["M", "texture"]},
        "addresses.json": addresses,
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
                }
            ],
        },
        "seeds.json": {"seeds": {SETPOINT: {"nominal": 5.0}}},
        "scenarios.json": {"scenarios": [{"name": "nominal"}]},
    }
    view = path / "simulator"
    view.mkdir(parents=True, exist_ok=True)
    for name, document in documents.items():
        (view / name).write_text(
            json.dumps({"schema": SCHEMAS[name], **document}), encoding="utf-8"
        )
    return view


def _composite(path: Path, clock: _Clock, **view: Any) -> Composite:
    return Composite(_view(path, **view), state_dir=path / "state", clock=clock, model_log=False)


@pytest.mark.parametrize(
    ("address", "kind"),
    [(HEARTBEAT, IntVariable), (UPTIME, ScalarVariable), (ENGINEER, StrVariable)],
)
def test_each_address_is_a_read_only_variable_of_its_type(
    tmp_path: Path, address: str, kind: type
) -> None:
    variable = _composite(tmp_path, _Clock()).supported_variables[address]

    assert isinstance(variable, kind)
    assert variable.read_only is True


def test_every_listed_model_is_served_the_texture_included(tmp_path: Path) -> None:
    composite = _composite(tmp_path, _Clock())

    assert {*TEXTURE_HEALTH, HEARTBEAT, UPTIME, ENGINEER} <= set(composite.supported_variables)
    assert composite.get(TEXTURE_HEALTH[0]) == 0


def test_heartbeat_and_uptime_advance_on_the_injected_clock(tmp_path: Path) -> None:
    clock = _Clock()
    composite = _composite(tmp_path, clock)

    assert composite.get([HEARTBEAT, UPTIME]) == {HEARTBEAT: 0, UPTIME: 0.0}
    clock.now += 2.5
    assert composite.get([HEARTBEAT, UPTIME]) == {HEARTBEAT: 2, UPTIME: 2.5}
    clock.now += 1.0
    assert composite.get([HEARTBEAT, UPTIME]) == {HEARTBEAT: 3, UPTIME: 3.5}


def test_uptime_is_numeric_seconds_and_heartbeat_a_counter(tmp_path: Path) -> None:
    clock = _Clock()
    composite = _composite(tmp_path, clock)
    clock.now += 61.25

    assert type(composite.get(UPTIME)) is float
    assert type(composite.get(HEARTBEAT)) is int


def test_a_clock_stepped_back_reads_zero_not_a_negative_uptime(tmp_path: Path) -> None:
    clock = _Clock()
    composite = _composite(tmp_path, clock)
    clock.now -= 30.0

    assert composite.get([HEARTBEAT, UPTIME]) == {HEARTBEAT: 0, UPTIME: 0.0}


def test_engineer_is_a_fixed_string(tmp_path: Path) -> None:
    clock = _Clock()
    composite = _composite(tmp_path, clock)
    first = composite.get(ENGINEER)
    clock.now += 10.0

    assert isinstance(first, str) and first
    assert composite.get(ENGINEER) == first


def test_held_reads_the_same_values(tmp_path: Path) -> None:
    clock = _Clock()
    composite = _composite(tmp_path, clock)
    clock.now += 4.0

    assert composite.held([HEARTBEAT, UPTIME]) == {HEARTBEAT: 4, UPTIME: 4.0}


def test_a_reset_does_not_restart_the_clock(tmp_path: Path) -> None:
    clock = _Clock()
    composite = _composite(tmp_path, clock)
    clock.now += 7.0

    composite.reset()

    assert composite.get(UPTIME) == 7.0


@pytest.mark.parametrize("address", [HEARTBEAT, UPTIME, ENGINEER])
def test_a_write_is_refused_by_the_read_only_check(tmp_path: Path, address: str) -> None:
    composite = _composite(tmp_path, _Clock())

    with pytest.raises(ReadOnlyError, match=address):
        composite.set({address: 1})


def test_a_no_deck_model_serves_all_three_without_touching_its_child(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = _Clock()
    composite = _composite(tmp_path, clock)

    def touched(*_args: Any) -> None:
        raise AssertionError("an IOC health read reached the child")

    monkeypatch.setattr(composite, "_plain_get", touched)
    monkeypatch.setattr(composite, "_read", touched)
    first = composite.get([HEARTBEAT, UPTIME, ENGINEER, STATUS])
    clock.now += 3.0
    second = composite.get([HEARTBEAT, UPTIME, ENGINEER, STATUS])

    assert composite._children["M"].model is None
    assert first[STATUS] == second[STATUS] == STATUS_NO_DECK
    assert second[HEARTBEAT] > first[HEARTBEAT]
    assert second[UPTIME] == first[UPTIME] + 3.0
    assert second[ENGINEER] == first[ENGINEER]


def test_a_no_deck_model_reports_no_undefined_health_address(tmp_path: Path) -> None:
    composite = _composite(tmp_path, _Clock())

    assert composite.output_severity([HEARTBEAT, UPTIME, ENGINEER]) == {}


def test_a_view_without_the_field_serves_none(tmp_path: Path) -> None:
    composite = _composite(tmp_path, _Clock(), health=None)

    assert set(composite.supported_variables) == {SETPOINT, STATUS}
    with pytest.raises(ValueError, match="not supported"):
        composite.get(HEARTBEAT)


def test_a_signal_this_composite_does_not_serve_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ViewSchemaError, match="T:SIM:M:TEMPERATURE"):
        _composite(tmp_path, _Clock(), health=("T:SIM:M:TEMPERATURE",))


@pytest.fixture
async def connector(tmp_path: Path):
    connected = VAInProcessConnector()
    await connected.connect({"simulator_view": str(_view(tmp_path)), "response_delay_ms": 0})
    yield connected
    await connected.disconnect()


@pytest.mark.parametrize("address", [HEARTBEAT, UPTIME, ENGINEER, *TEXTURE_HEALTH])
async def test_the_in_process_connector_serves_each_address(
    connector: VAInProcessConnector, address: str
) -> None:
    assert await connector.validate_channel(address) is True
    assert (await connector.read_channel(address)).value is not None


async def test_the_in_process_connector_reads_a_numeric_uptime(
    connector: VAInProcessConnector,
) -> None:
    assert isinstance((await connector.read_channel(UPTIME)).value, float)


@pytest.mark.parametrize("address", [HEARTBEAT, UPTIME, ENGINEER])
async def test_the_in_process_connector_refuses_a_write(
    connector: VAInProcessConnector, address: str
) -> None:
    result = await connector.write_channel(address, 1)

    assert result.outcome is WriteOutcome.REFUSED
