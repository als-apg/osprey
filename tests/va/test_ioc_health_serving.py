"""The serving side of the IOC health addresses, over a real composite and a hand-written view.

``periodic_addresses`` and the model RPC's surface read the view's optional
``ioc_health`` field through a guarded accessor, so a connectors release without
it serves none. A periodic pass publishes the advancing addresses; a write pass
leaves them for the next one.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from osprey.services.virtual_accelerator.serving.model_rpc import ModelRpcError
from osprey.services.virtual_accelerator.serving.model_surface import (
    SURFACE_SERVED,
    ModelSurface,
)
from osprey.services.virtual_accelerator.serving.runner_config import (
    apply_safety,
    ioc_health_addresses,
    periodic_addresses,
    setting_records,
)
from osprey_connectors.simulation.composite import Composite
from osprey_connectors.simulation.view import SCHEMAS, SimulatorView
from tests._simulator_view import view_channel

TOKEN = "s3cret"
SETPOINT = "T:SP"
STATUS = "T:SIM:M:STATUS"
HEARTBEAT = "T:SIM:M:HEARTBEAT"
UPTIME = "T:SIM:M:UPTIME"
ENGINEER = "T:SIM:M:ENGINEER"
HEALTH = (HEARTBEAT, UPTIME, ENGINEER)
T0 = 1_760_000_000.0


class _Clock:
    def __init__(self) -> None:
        self.now = T0

    def __call__(self) -> float:
        return self.now


class _OlderView:
    """A view as a connectors release without ``ioc_health_by_model`` presents it."""

    def __init__(self, view: SimulatorView) -> None:
        self._view = view

    def __getattr__(self, name: str) -> Any:
        if name == "ioc_health_by_model":
            raise AttributeError(name)
        return getattr(self._view, name)


def _view(path: Path, *, health: bool = True) -> SimulatorView:
    """A view of one texture setpoint and a physics model ``M`` with no deck."""
    addresses: dict[str, Any] = {"channels": [SETPOINT], "status": [STATUS]}
    if health:
        addresses["ioc_health"] = list(HEALTH)
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
            "channels": [view_channel(SETPOINT, role="setpoint", writable=True)],
        },
        "seeds.json": {"seeds": {SETPOINT: {"nominal": 5.0}}},
        "scenarios.json": {"scenarios": [{"name": "nominal"}]},
    }
    directory = path / "simulator"
    directory.mkdir(parents=True, exist_ok=True)
    for name, document in documents.items():
        (directory / name).write_text(
            json.dumps({"schema": SCHEMAS[name], **document}), encoding="utf-8"
        )
    return SimulatorView.open(directory)


def _composite(path: Path, view: SimulatorView, clock: _Clock) -> Composite:
    return Composite(view, state_dir=path / "state", clock=clock, model_log=False)


def _surface(view: Any, composite: Composite) -> ModelSurface:
    return ModelSurface.for_view(
        composite,
        view,
        instance="va-1",
        endpoint="va-1:5075",
        model_write_token=TOKEN,
        failed_pass_tolerance=3,
        clock=lambda: T0,
    )


def _served(surface: ModelSurface) -> list[str]:
    return [e["name"] for e in surface.info()["variables"] if e["surface"] == SURFACE_SERVED]


def test_heartbeat_and_uptime_are_periodic_and_engineer_is_not(tmp_path: Path) -> None:
    assert periodic_addresses(_view(tmp_path)) == {HEARTBEAT, UPTIME}


def test_a_view_without_the_accessor_names_no_periodic_health_address(tmp_path: Path) -> None:
    assert periodic_addresses(_OlderView(_view(tmp_path))) == frozenset()


def test_a_view_without_the_field_names_no_periodic_health_address(tmp_path: Path) -> None:
    assert periodic_addresses(_view(tmp_path, health=False)) == frozenset()


def test_ioc_health_addresses_are_every_health_address_of_the_view(tmp_path: Path) -> None:
    assert ioc_health_addresses(_view(tmp_path)) == {HEARTBEAT, UPTIME, ENGINEER}


def test_a_view_without_the_accessor_names_no_ioc_health_address(tmp_path: Path) -> None:
    assert ioc_health_addresses(_OlderView(_view(tmp_path))) == frozenset()


def test_a_view_without_the_field_names_no_ioc_health_address(tmp_path: Path) -> None:
    assert ioc_health_addresses(_view(tmp_path, health=False)) == frozenset()


def test_the_engineer_variable_is_text_and_the_counters_are_not(tmp_path: Path) -> None:
    from lume.variables import StrVariable

    view = _view(tmp_path)
    variables = _composite(tmp_path, view, _Clock()).supported_variables

    assert isinstance(variables[ENGINEER], StrVariable)
    assert not isinstance(variables[HEARTBEAT], StrVariable)
    assert not isinstance(variables[UPTIME], StrVariable)


def test_no_setting_reads_back_a_health_address(tmp_path: Path) -> None:
    assert not any(set(HEALTH) & records for records in setting_records(_view(tmp_path)).values())


def test_each_health_address_is_served_read_only_on_the_wire(tmp_path: Path) -> None:
    config = {"variables": {address: {"name": address, "pv": address} for address in HEALTH}}

    safe = apply_safety(config, _view(tmp_path))

    assert {safe["variables"][address]["mode"] for address in HEALTH} == {"ro"}


def test_info_lists_the_health_addresses_after_the_status(tmp_path: Path) -> None:
    view = _view(tmp_path)

    surface = _surface(view, _composite(tmp_path, view, _Clock()))

    assert _served(surface) == [SETPOINT, STATUS, *HEALTH]


def test_info_declares_each_health_address_read_only_with_uptime_in_seconds(
    tmp_path: Path,
) -> None:
    view = _view(tmp_path)

    entries = {
        entry["name"]: entry
        for entry in _surface(view, _composite(tmp_path, view, _Clock())).info()["variables"]
    }

    assert all(entries[address]["read_only"] for address in HEALTH)
    assert entries[UPTIME]["unit"] == "s"


def test_get_reads_each_health_address_as_served(tmp_path: Path) -> None:
    view = _view(tmp_path)
    clock = _Clock()
    surface = _surface(view, _composite(tmp_path, view, clock))
    clock.now += 5.0

    values = surface.get(HEALTH)

    assert values[HEARTBEAT] == 5
    assert values[UPTIME] == 5.0
    assert isinstance(values[ENGINEER], str)


def test_the_model_rpc_refuses_a_write_to_a_health_address(tmp_path: Path) -> None:
    view = _view(tmp_path)
    surface = _surface(view, _composite(tmp_path, view, _Clock()))

    with pytest.raises(ModelRpcError, match=HEARTBEAT):
        surface.set({HEARTBEAT: 1}, TOKEN)


def test_info_on_a_view_without_the_accessor_lists_no_health_address(tmp_path: Path) -> None:
    view = _view(tmp_path, health=False)

    surface = _surface(_OlderView(view), _composite(tmp_path, view, _Clock()))

    assert _served(surface) == [SETPOINT, STATUS]


def test_get_on_a_view_without_the_accessor_refuses_a_health_address(tmp_path: Path) -> None:
    view = _view(tmp_path, health=False)
    surface = _surface(_OlderView(view), _composite(tmp_path, view, _Clock()))

    assert surface.get([STATUS]) == {STATUS: "no deck"}
    with pytest.raises(ModelRpcError):
        surface.get([HEARTBEAT])


def test_two_periodic_passes_publish_an_advanced_heartbeat(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip(
        "lume_pva_apg", reason="the served VA runs on lume-pva-apg, which ships for Linux only"
    )
    from lume_pva_apg.runner import Runner

    from osprey.services.virtual_accelerator.serving.runner import ModelRunner

    view = _view(tmp_path)
    clock = _Clock()
    composite = _composite(tmp_path, view, clock)
    posted: list[dict[str, Any]] = []
    monkeypatch.setattr(
        Runner, "_cycle_output_names", lambda _self: list(composite.supported_variables)
    )
    monkeypatch.setattr(
        Runner,
        "_run_cycle",
        lambda self, _item: self._post_outputs(composite.get(self._cycle_output_names()), T0),
    )
    monkeypatch.setattr(Runner, "_post_outputs", lambda _self, values, _ts: posted.append(values))
    runner = object.__new__(ModelRunner)
    runner.model = composite
    runner._cycle_requested_names = ()
    runner._timed_names = frozenset()
    runner._needs_timed = False
    runner._periodic = periodic_addresses(view)
    runner._setting_records = setting_records(view)
    runner._served = {}
    runner._udf = set()
    runner._evaluate_severity = lambda _names: set()

    ModelRunner._run_cycle(runner, {"values": {}, "done": []})
    clock.now += 2.0
    ModelRunner._run_cycle(runner, {"values": {}, "done": []})
    clock.now += 2.0
    ModelRunner._run_cycle(runner, {"values": {SETPOINT: 1.0}, "done": []})

    first, second, written = posted
    assert second[HEARTBEAT] > first[HEARTBEAT]
    assert second[UPTIME] > first[UPTIME]
    assert HEARTBEAT not in written
    assert UPTIME not in written


def test_a_text_health_address_is_a_ca_string_and_the_rest_keep_their_spec(
    tmp_path: Path,
) -> None:
    pytest.importorskip(
        "lume_pva_apg", reason="the served VA runs on lume-pva-apg, which ships for Linux only"
    )
    from lume_pva_apg.runner import Runner
    from lume_pva_apg.variables import find_variable_handler

    from osprey.services.virtual_accelerator.serving.runner import ModelRunner

    view = _view(tmp_path)
    variables = _composite(tmp_path, view, _Clock()).supported_variables

    def runner(cls: type) -> Any:
        made = object.__new__(cls)
        made.pvdb = {}
        made.ca_pvs = {}
        made.supports_ca = True
        made.supports_pva = False
        made._pv_meta = {}
        made.pv_handlers = {}
        made._ioc_health = frozenset({ENGINEER, HEARTBEAT})
        return made

    served, plain = runner(ModelRunner), runner(Runner)
    for address in (ENGINEER, HEARTBEAT, STATUS):
        var = variables[address]
        handler = find_variable_handler(type(var))
        ModelRunner._add_pv(served, address, var, True, "", handler)
        Runner._add_pv(plain, address, var, True, "", handler)

    assert served.pvdb[ENGINEER] == {"type": "string", "asyn": True}
    assert served.pvdb[HEARTBEAT] == plain.pvdb[HEARTBEAT]
    assert served.pvdb[STATUS]["type"] == "char"
